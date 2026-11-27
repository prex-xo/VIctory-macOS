//! Turns a title's image into something the Windows loader maps but does
//! not run. Pure byte manipulation, so it is testable on any host.

use victory_pe::layout::{self, dir, dll_flags, file_flags};
use victory_pe::{ImageInfo, PeError};

#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    #[error(transparent)]
    Pe(#[from] PeError),
    #[error("cannot patch image: {0}")]
    Patch(&'static str),
}

/// A patched copy of an image plus everything we need from the original.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// Parsed from the *original* bytes: imports, TLS callbacks and entry
    /// point as the title expects them.
    pub info: ImageInfo,
    /// The patched bytes to hand to the Windows loader.
    pub bytes: Vec<u8>,
    /// BLAKE3 of the original bytes: identifies the exact build.
    pub original_hash: blake3::Hash,
}

impl PreparedImage {
    /// File name for the patched copy, unique per original build.
    pub fn file_name(&self) -> String {
        format!("victory-image-{}.dll", &self.original_hash.to_hex()[..16])
    }
}

/// Produces the patched copy. The changes are:
///
/// * `IMAGE_FILE_DLL` set, entry point cleared: the OS maps it like a DLL
///   and calls nothing;
/// * import directory points to its empty terminator and the bound-import
///   directory is cleared: we bind the original slots using `info.imports`;
/// * TLS callback pointer cleared: the OS still allocates static TLS for
///   the image, but callbacks only run when we call them (after binding);
///   its base relocation is made a no-op so the 0 survives a rebase;
/// * `FORCE_INTEGRITY` cleared, checksum zeroed: the copy is unsigned.
pub fn prepare(original: &[u8]) -> Result<PreparedImage, PrepareError> {
    let info = ImageInfo::parse(original)?;
    let mut bytes = original.to_vec();

    let fh = layout::file_header_offset(&bytes).ok_or(PrepareError::Patch("no file header"))?;
    let opt =
        layout::optional_header_offset(&bytes).ok_or(PrepareError::Patch("no optional header"))?;

    let characteristics = info.characteristics | file_flags::DLL;
    write_u16(
        &mut bytes,
        fh + layout::FILE_HEADER_CHARACTERISTICS,
        characteristics,
    )?;
    write_u32(&mut bytes, opt + layout::OPT_ADDRESS_OF_ENTRY_POINT, 0)?;
    write_u32(&mut bytes, opt + layout::OPT_CHECKSUM, 0)?;
    let dll_characteristics = info.dll_characteristics & !dll_flags::FORCE_INTEGRITY;
    write_u16(
        &mut bytes,
        opt + layout::OPT_DLL_CHARACTERISTICS,
        dll_characteristics,
    )?;

    // Windows 10 assigns static TLS to the prepared DLL when its empty
    // import table is present. The parser has already walked one descriptor
    // per module up to the null terminator, regardless of the recorded Size.
    if let Some(at) = layout::data_directory_offset(&bytes, dir::IMPORT) {
        let rva = layout::read_u32(&bytes, at).ok_or(PrepareError::Patch("no import RVA"))?;
        if rva != 0 {
            let terminator = u32::try_from(info.imports.len())
                .ok()
                .and_then(|count| count.checked_mul(IMPORT_DESCRIPTOR_SIZE))
                .and_then(|offset| rva.checked_add(offset))
                .ok_or(PrepareError::Patch("import terminator RVA overflow"))?;
            write_u32(&mut bytes, at, terminator)?;
            write_u32(&mut bytes, at + 4, IMPORT_DESCRIPTOR_SIZE)?;
        }
    }
    if let Some(at) = layout::data_directory_offset(&bytes, dir::BOUND_IMPORT) {
        write(&mut bytes, at, &[0; layout::DATA_DIRECTORY_SIZE])?;
    }

    if let Some(tls) = &info.tls
        && tls.callbacks_va != 0
    {
        let at = info
            .rva_to_file_offset(tls.directory_rva)
            .ok_or(PrepareError::Patch("TLS directory is not file-backed"))?;
        write_u64(
            &mut bytes,
            at as usize + layout::TLS64_ADDRESS_OF_CALLBACKS,
            0,
        )?;
        // The field's base relocation would turn the 0 into `delta` whenever
        // the loader maps the copy away from its preferred base (Wine always
        // does for DYNAMIC_BASE images), and the loader then reads callbacks
        // from there. Make that entry a no-op.
        let field_rva = tls
            .directory_rva
            .checked_add(layout::TLS64_ADDRESS_OF_CALLBACKS as u32)
            .ok_or(PrepareError::Patch("TLS callback field RVA overflow"))?;
        for (entry, target) in dir64_relocations(&bytes, &info) {
            if target == field_rva {
                write_u16(&mut bytes, entry, 0)?; // IMAGE_REL_BASED_ABSOLUTE
            }
        }
    }

    Ok(PreparedImage {
        info,
        bytes,
        original_hash: blake3::hash(original),
    })
}

/// Size of an IMAGE_IMPORT_DESCRIPTOR in the PE file.
const IMPORT_DESCRIPTOR_SIZE: u32 = 20;

/// Every `IMAGE_REL_BASED_DIR64` base relocation: (file offset of its entry,
/// RVA it patches). Empty if the image has no readable relocation table.
fn dir64_relocations(bytes: &[u8], info: &ImageInfo) -> Vec<(usize, u32)> {
    const DIR64: u16 = 10;
    let mut out = Vec::new();
    let Some(d) = layout::data_directory_offset(bytes, dir::BASERELOC) else {
        return out;
    };
    let (Some(rva), Some(size)) = (layout::read_u32(bytes, d), layout::read_u32(bytes, d + 4))
    else {
        return out;
    };
    let Some(start) = info.rva_to_file_offset(rva).filter(|_| size != 0) else {
        return out;
    };
    let (mut at, end) = (start as usize, start as usize + size as usize);
    while at + 8 <= end {
        let (Some(page), Some(block)) =
            (layout::read_u32(bytes, at), layout::read_u32(bytes, at + 4))
        else {
            break;
        };
        let block = block as usize;
        if block < 8 || at + block > end {
            break;
        }
        // An odd block size ends in a byte that is not a whole entry: entry
        // reads stay within the block's declared end.
        let block_end = at + block;
        let mut entry = at + 8;
        while entry + 2 <= block_end {
            let Some(e) = bytes.get(entry..entry + 2) else {
                break;
            };
            let e = u16::from_le_bytes([e[0], e[1]]);
            if e >> 12 == DIR64 {
                // A page RVA near the 4 GiB limit can overflow with the
                // entry's offset; like any malformed entry, skip it.
                if let Some(target) = page.checked_add(u32::from(e & 0xFFF)) {
                    out.push((entry, target));
                }
            }
            entry += 2;
        }
        at += block;
    }
    out
}

fn write_u16(bytes: &mut [u8], at: usize, v: u16) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write_u32(bytes: &mut [u8], at: usize, v: u32) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write_u64(bytes: &mut [u8], at: usize, v: u64) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write(bytes: &mut [u8], at: usize, v: &[u8]) -> Result<(), PrepareError> {
    bytes
        .get_mut(at..at + v.len())
        .ok_or(PrepareError::Patch("header field outside of file"))?
        .copy_from_slice(v);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `ImageInfo` with one section mapping `rva` to `file_offset`.
    fn info_with_section(rva: u32, file_offset: u32, file_size: u32) -> ImageInfo {
        ImageInfo {
            machine: victory_pe::MACHINE_AMD64,
            characteristics: 0,
            timestamp: 0,
            subsystem: 0,
            dll_characteristics: 0,
            image_base: 0,
            size_of_image: 0,
            entry_rva: 0,
            stack_reserve: 0,
            stack_commit: 0,
            sections: vec![victory_pe::SectionInfo {
                name: ".sec".into(),
                rva,
                virtual_size: file_size,
                file_offset,
                file_size,
                characteristics: 0,
            }],
            imports: Vec::new(),
            delay_imports: Vec::new(),
            exports: Vec::new(),
            tls: None,
            has_relocations: true,
            has_exception_table: false,
            has_load_config: false,
            codeview: None,
        }
    }

    #[test]
    fn dir64_relocations_skips_entries_past_the_4_gib_limit() {
        // A header whose BASERELOC directory points at RVA 0x1000, 24 bytes,
        // mapped to file offset 0x200 by the section above.
        let mut bytes = vec![0u8; 0x300];
        bytes[..2].copy_from_slice(b"MZ");
        bytes[layout::DOS_E_LFANEW..layout::DOS_E_LFANEW + 4]
            .copy_from_slice(&0x40u32.to_le_bytes());
        let opt = layout::optional_header_offset(&bytes).unwrap();
        let count = opt + layout::OPT_NUMBER_OF_RVA_AND_SIZES;
        bytes[count..count + 4].copy_from_slice(&16u32.to_le_bytes());
        let d = layout::data_directory_offset(&bytes, dir::BASERELOC).unwrap();
        bytes[d..d + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        bytes[d + 4..d + 8].copy_from_slice(&24u32.to_le_bytes());
        let info = info_with_section(0x1000, 0x200, 0x100);
        // Block 1: page 0x1000 with one DIR64 entry at offset 0x123 and one
        // ABSOLUTE entry (not collected).
        let at = 0x200;
        bytes[at..at + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        bytes[at + 4..at + 8].copy_from_slice(&12u32.to_le_bytes());
        bytes[at + 8..at + 10].copy_from_slice(&0xA123u16.to_le_bytes());
        bytes[at + 10..at + 12].copy_from_slice(&0x0123u16.to_le_bytes());
        // Block 2: page u32::MAX - 0x10; the 0x10 offset lands exactly on
        // u32::MAX, the 0xFFF offset overflows and is skipped (a panic in
        // debug builds before).
        let at = 0x20C;
        bytes[at..at + 4].copy_from_slice(&(u32::MAX - 0x10).to_le_bytes());
        bytes[at + 4..at + 8].copy_from_slice(&12u32.to_le_bytes());
        bytes[at + 8..at + 10].copy_from_slice(&0xA010u16.to_le_bytes());
        bytes[at + 10..at + 12].copy_from_slice(&0xAFFFu16.to_le_bytes());
        assert_eq!(
            dir64_relocations(&bytes, &info),
            vec![(0x208, 0x1123), (0x214, u32::MAX)]
        );
    }

    #[test]
    fn dir64_relocations_stops_at_an_odd_blocks_end() {
        // BASERELOC at RVA 0x1000, 21 bytes, mapped to file offset 0x200.
        // Block 1 holds one DIR64 entry; block 2 has an odd size of 9, so its
        // last byte is not an entry even though the two bytes at 0x214 would
        // decode as DIR64 — the second lies past the block's declared end.
        let mut bytes = vec![0u8; 0x300];
        bytes[..2].copy_from_slice(b"MZ");
        bytes[layout::DOS_E_LFANEW..layout::DOS_E_LFANEW + 4]
            .copy_from_slice(&0x40u32.to_le_bytes());
        let opt = layout::optional_header_offset(&bytes).unwrap();
        let count = opt + layout::OPT_NUMBER_OF_RVA_AND_SIZES;
        bytes[count..count + 4].copy_from_slice(&16u32.to_le_bytes());
        let d = layout::data_directory_offset(&bytes, dir::BASERELOC).unwrap();
        bytes[d..d + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        bytes[d + 4..d + 8].copy_from_slice(&21u32.to_le_bytes());
        let info = info_with_section(0x1000, 0x200, 0x100);
        // Block 1: page 0x1000, one DIR64 entry at offset 0x123, one ABSOLUTE.
        let at = 0x200;
        bytes[at..at + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        bytes[at + 4..at + 8].copy_from_slice(&12u32.to_le_bytes());
        bytes[at + 8..at + 10].copy_from_slice(&0xA123u16.to_le_bytes());
        bytes[at + 10..at + 12].copy_from_slice(&0x0123u16.to_le_bytes());
        // Block 2: page 0x2000, odd size 9; the byte at 0x214 pairs with the
        // byte at 0x215 (past the block) into what looks like DIR64 0x123.
        let at = 0x20C;
        bytes[at..at + 4].copy_from_slice(&0x2000u32.to_le_bytes());
        bytes[at + 4..at + 8].copy_from_slice(&9u32.to_le_bytes());
        bytes[at + 8] = 0x23;
        bytes[at + 9] = 0xA1;
        assert_eq!(dir64_relocations(&bytes, &info), vec![(0x208, 0x1123)]);
    }

    /// A minimal PE32+ image whose single section sits at RVA 0xFFFF_F000
    /// (file offset 0x200, 0x2000 bytes) and whose TLS directory is at
    /// `directory_rva`, with an empty callback array at RVA 0xFFFF_FF00.
    /// Parses with `ImageInfo::parse` on any host.
    fn synthetic_tls_image(directory_rva: u32) -> Vec<u8> {
        let mut d = vec![0u8; 0x200];
        d[..2].copy_from_slice(b"MZ");
        d[layout::DOS_E_LFANEW..layout::DOS_E_LFANEW + 4].copy_from_slice(&0x40u32.to_le_bytes());
        d[0x40..0x44].copy_from_slice(b"PE\0\0");
        d[0x44..0x46].copy_from_slice(&victory_pe::MACHINE_AMD64.to_le_bytes());
        d[0x46..0x48].copy_from_slice(&1u16.to_le_bytes()); // one section
        d[0x54..0x56].copy_from_slice(&240u16.to_le_bytes()); // optional header size
        d[0x56..0x58].copy_from_slice(&0x2022u16.to_le_bytes()); // characteristics
        let opt = 0x58;
        d[opt..opt + 2].copy_from_slice(&0x20Bu16.to_le_bytes()); // PE32+
        d[opt + 24..opt + 32].copy_from_slice(&0x1_8000_0000u64.to_le_bytes()); // image base
        d[opt + 32..opt + 36].copy_from_slice(&0x1000u32.to_le_bytes()); // section alignment
        d[opt + 36..opt + 40].copy_from_slice(&0x200u32.to_le_bytes()); // file alignment
        d[opt + 56..opt + 60].copy_from_slice(&0x2000u32.to_le_bytes()); // size of image
        d[opt + 60..opt + 64].copy_from_slice(&0x200u32.to_le_bytes()); // size of headers
        let count = opt + layout::OPT_NUMBER_OF_RVA_AND_SIZES;
        d[count..count + 4].copy_from_slice(&16u32.to_le_bytes());
        let tls_dir = opt + layout::OPT_DATA_DIRECTORY + 9 * layout::DATA_DIRECTORY_SIZE;
        d[tls_dir..tls_dir + 4].copy_from_slice(&directory_rva.to_le_bytes());
        d[tls_dir + 4..tls_dir + 8].copy_from_slice(&40u32.to_le_bytes());
        let sec = opt + 240;
        d[sec..sec + 4].copy_from_slice(b".tls");
        d[sec + 8..sec + 12].copy_from_slice(&0x2000u32.to_le_bytes()); // virtual size
        d[sec + 12..sec + 16].copy_from_slice(&0xFFFF_F000u32.to_le_bytes()); // RVA
        d[sec + 16..sec + 20].copy_from_slice(&0x2000u32.to_le_bytes()); // raw size
        d[sec + 20..sec + 24].copy_from_slice(&0x200u32.to_le_bytes()); // raw offset
        d[sec + 36..sec + 40].copy_from_slice(&0x4000_0040u32.to_le_bytes()); // read, data
        d.resize(0x2200, 0);
        // The TLS directory's AddressOfCallBacks points at an empty array
        // at RVA 0xFFFF_FF00 (image base 0x1_8000_0000).
        let at = 0x200 + (directory_rva - 0xFFFF_F000) as usize;
        let cb = at + layout::TLS64_ADDRESS_OF_CALLBACKS;
        d[cb..cb + 8].copy_from_slice(&(0x1_8000_0000u64 + 0xFFFF_FF00).to_le_bytes());
        let array = 0x200 + (0xFFFF_FF00u32 - 0xFFFF_F000) as usize;
        d[array..array + 8].copy_from_slice(&0u64.to_le_bytes());
        d
    }

    #[test]
    fn a_tls_directory_past_the_4_gib_limit_is_an_error_not_a_panic() {
        // directory_rva + TLS64_ADDRESS_OF_CALLBACKS overflows u32: a panic
        // in debug builds, a wrapped RVA in release ones, before. The parse
        // now refuses a directory outside the image, so `prepare` never
        // gets to the addition.
        let image = synthetic_tls_image(0xFFFF_FFF0);
        assert!(ImageInfo::parse(&image).is_err());
        assert!(prepare(&image).is_err());
    }

    #[test]
    #[cfg_attr(not(windows), ignore = "reads its own test binary as a PE file")]
    fn patches_own_test_binary() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let prepared = prepare(&original).unwrap();
        assert!(!prepared.info.is_dll(), "info describes the original");
        assert!(prepared.info.import_count() > 0);

        let patched = ImageInfo::parse(&prepared.bytes).unwrap();
        assert!(patched.is_dll());
        assert_eq!(patched.entry_rva, 0);
        assert!(patched.imports.is_empty());
        assert_eq!(patched.dll_characteristics & dll_flags::FORCE_INTEGRITY, 0);
        if let Some(tls) = &patched.tls {
            assert_eq!(tls.callbacks_va, 0);
            let field = tls.directory_rva + layout::TLS64_ADDRESS_OF_CALLBACKS as u32;
            assert!(
                dir64_relocations(&prepared.bytes, &patched)
                    .iter()
                    .all(|&(_, target)| target != field),
                "the cleared callback pointer must not be relocated"
            );
        }
        // Nothing else moved: same sections, same size.
        assert_eq!(patched.sections.len(), prepared.info.sections.len());
        assert_eq!(prepared.bytes.len(), original.len());
    }

    #[test]
    #[cfg_attr(not(windows), ignore = "reads its own test binary as a PE file")]
    fn keeps_iat_slots_when_import_size_excludes_the_terminator() {
        let mut original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let info = ImageInfo::parse(&original).unwrap();
        let directory = layout::data_directory_offset(&original, dir::IMPORT).unwrap();
        // This linker layout counts the descriptors for real modules only.
        let descriptor_bytes = u32::try_from(info.imports.len()).unwrap() * IMPORT_DESCRIPTOR_SIZE;
        write_u32(&mut original, directory + 4, descriptor_bytes).unwrap();
        let prepared = prepare(&original).unwrap();
        let empty = layout::read_u32(&prepared.bytes, directory).unwrap();
        assert_eq!(
            empty,
            layout::read_u32(&original, directory).unwrap() + descriptor_bytes
        );
        assert_eq!(
            layout::read_u32(&prepared.bytes, directory + 4),
            Some(IMPORT_DESCRIPTOR_SIZE)
        );
        let at = info.rva_to_file_offset(empty).unwrap() as usize;
        assert_eq!(
            &prepared.bytes[at..at + IMPORT_DESCRIPTOR_SIZE as usize],
            &[0; 20]
        );
        let iat = layout::data_directory_offset(&original, dir::IAT).unwrap();
        assert_eq!(
            &prepared.bytes[iat..iat + layout::DATA_DIRECTORY_SIZE],
            &original[iat..iat + layout::DATA_DIRECTORY_SIZE]
        );
        assert_eq!(prepared.info.import_count(), info.import_count());
        for module in &prepared.info.imports {
            for entry in &module.entries {
                let at = info.rva_to_file_offset(entry.iat_slot_rva).unwrap() as usize;
                assert_eq!(&prepared.bytes[at..at + 8], &original[at..at + 8]);
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_assigns_static_tls_to_a_prepared_dll() {
        // Model a title DLL using our own linked image. Loading a DLL leaves
        // its assigned index intact; the executable-specific redirect resets it.
        let mut original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let flags =
            layout::file_header_offset(&original).unwrap() + layout::FILE_HEADER_CHARACTERISTICS;
        let characteristics =
            ImageInfo::parse(&original).unwrap().characteristics | file_flags::DLL;
        write_u16(&mut original, flags, characteristics).unwrap();
        let prepared = prepare(&original).unwrap();
        let info = &prepared.info;
        let tls = info.tls.as_ref().expect("the test binary uses static TLS");
        let directory = info.rva_to_file_offset(tls.directory_rva).unwrap() as usize;
        let index_va = layout::read_u64(&original, directory + 16).unwrap();
        let index_rva = index_va.checked_sub(info.image_base).unwrap();
        assert!(index_rva + 4 <= u64::from(info.size_of_image));

        let folder = tempfile::tempdir().unwrap();
        let image = crate::load(&prepared, folder.path()).unwrap();
        // SAFETY: the mapped test image contains the checked DWORD above.
        let assigned =
            unsafe { ((image.base() + index_rva as usize) as *const u32).read_unaligned() };
        // SAFETY: the prepared DLL has never been attached or given to a caller.
        unsafe { image.unload() };
        assert_ne!(
            assigned, 0,
            "the process executable already owns TLS slot zero"
        );
    }

    #[test]
    #[cfg_attr(not(windows), ignore = "reads its own test binary as a PE file")]
    fn file_name_is_stable_per_build() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let a = prepare(&original).unwrap();
        let b = prepare(&original).unwrap();
        assert_eq!(a.file_name(), b.file_name());
    }
}
