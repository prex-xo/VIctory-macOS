//! World-frame-indexed input recording and deterministic replay.
//!
//! Activated only by environment variables:
//! - `VICTORY_INPUT_RECORD=<path.jsonl>`: records `PadState` and `KeyState`
//!   changes tagged with `victory_gpu::bench::world_frame()`.
//! - `VICTORY_INPUT_REPLAY=<path.jsonl>`: replays recorded traces or
//!   declarative range lines (`{"from":0,"to":120,"ly":1.0,"rt":0.8}`) indexed
//!   by `world_frame()`. While `world_frame() == 0` (menus/loading screens),
//!   gamepad sticks/buttons stay neutral while keyboard and scripted keys pass
//!   through so menu navigation (`VICTORY_KEYS`) still reaches the world; once
//!   `world_frame() >= 1`, the replay script takes over deterministically.
//!
//! If both are set, the recording must use a different file from the replay.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, ErrorKind, Write};
use std::path::Path;
use std::time::Instant;

use super::gameinput::{KeyState, PadState, Source};

const PADS: usize = 4;

#[derive(Clone, Debug, PartialEq)]
struct TracePadEntry {
    wframe: u64,
    slot: usize,
    connected: bool,
    pad: PadState,
}

#[derive(Clone, Debug, PartialEq)]
struct TraceKeysEntry {
    wframe: u64,
    keys: Vec<KeyState>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct RangeEntry {
    from: u64,
    to: u64,
    slot: usize,
    buttons: Option<u32>,
    lt: Option<f32>,
    rt: Option<f32>,
    lx: Option<f32>,
    ly: Option<f32>,
    rx: Option<f32>,
    ry: Option<f32>,
    keys: Vec<KeyState>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Script {
    pad_traces: [Vec<TracePadEntry>; PADS],
    key_traces: Vec<TraceKeysEntry>,
    ranges: Vec<RangeEntry>,
}

impl Script {
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        let mut script = Self::default();
        for (line_no, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            let obj =
                parse_json_object(line).map_err(|e| format!("line {}: {e}", line_no + 1))?;
            if obj.contains_key("from") || obj.contains_key("to") {
                let from = obj_u64(&obj, "from")?.unwrap_or(0);
                let to = obj_u64(&obj, "to")?.unwrap_or(u64::MAX);
                if to < from {
                    return Err(format!(
                        "line {}: `to` ({to}) < `from` ({from})",
                        line_no + 1
                    ));
                }
                let slot = obj_u64(&obj, "slot")?.unwrap_or(0) as usize;
                if slot >= PADS {
                    return Err(format!(
                        "line {}: pad slot {slot} out of range",
                        line_no + 1
                    ));
                }
                let keys = obj_keys(&obj, "keys")?;
                script.ranges.push(RangeEntry {
                    from,
                    to,
                    slot,
                    buttons: obj_u64(&obj, "buttons")?.map(|v| v as u32),
                    lt: obj_f32(&obj, "lt")?,
                    rt: obj_f32(&obj, "rt")?,
                    lx: obj_f32(&obj, "lx")?,
                    ly: obj_f32(&obj, "ly")?,
                    rx: obj_f32(&obj, "rx")?,
                    ry: obj_f32(&obj, "ry")?,
                    keys,
                });
                continue;
            }

            let wframe = obj_u64(&obj, "wframe")?
                .ok_or_else(|| format!("line {}: missing `wframe` or `from`/`to`", line_no + 1))?;
            if let Some(keys_val) = obj.get("keys")
                && !obj.contains_key("slot")
                && !obj.contains_key("connected")
                && !obj.contains_key("buttons")
            {
                let keys = parse_keys_value(keys_val)
                    .map_err(|e| format!("line {}: {e}", line_no + 1))?;
                script.key_traces.push(TraceKeysEntry { wframe, keys });
                continue;
            }

            let slot = obj_u64(&obj, "slot")?.unwrap_or(0) as usize;
            if slot >= PADS {
                return Err(format!(
                    "line {}: pad slot {slot} out of range",
                    line_no + 1
                ));
            }
            let connected = obj_bool(&obj, "connected")?.unwrap_or(true);
            let pad = PadState {
                buttons: obj_u64(&obj, "buttons")?.unwrap_or(0) as u32,
                left_trigger: obj_f32(&obj, "lt")?.unwrap_or(0.0),
                right_trigger: obj_f32(&obj, "rt")?.unwrap_or(0.0),
                left_x: obj_f32(&obj, "lx")?.unwrap_or(0.0),
                left_y: obj_f32(&obj, "ly")?.unwrap_or(0.0),
                right_x: obj_f32(&obj, "rx")?.unwrap_or(0.0),
                right_y: obj_f32(&obj, "ry")?.unwrap_or(0.0),
            };
            script.pad_traces[slot].push(TracePadEntry {
                wframe,
                slot,
                connected,
                pad,
            });
            if obj.contains_key("keys") {
                let keys = obj_keys(&obj, "keys")?;
                script.key_traces.push(TraceKeysEntry { wframe, keys });
            }
        }
        for entries in &mut script.pad_traces {
            entries.sort_by_key(|e| e.wframe);
        }
        script.key_traces.sort_by_key(|e| e.wframe);
        Ok(script)
    }

    pub(crate) fn pad_at(&self, wframe: u64, slot: usize) -> Option<PadState> {
        if slot >= PADS {
            return None;
        }
        // Hold neutral during loading screens (`wframe == 0`), keeping slot 0
        // connected so the title does not show a disconnect prompt.
        if wframe == 0 {
            return (slot == 0).then(PadState::default);
        }
        let mut state = match self.pad_traces[slot].partition_point(|e| e.wframe <= wframe) {
            0 => (slot == 0).then(PadState::default),
            idx => {
                let entry = &self.pad_traces[slot][idx - 1];
                entry.connected.then_some(entry.pad)
            }
        };
        for range in &self.ranges {
            if range.slot == slot && range.from <= wframe && wframe <= range.to {
                let pad = state.get_or_insert_with(PadState::default);
                if let Some(b) = range.buttons {
                    pad.buttons |= b;
                }
                if let Some(v) = range.lt {
                    pad.left_trigger = v;
                }
                if let Some(v) = range.rt {
                    pad.right_trigger = v;
                }
                if let Some(v) = range.lx {
                    pad.left_x = v;
                }
                if let Some(v) = range.ly {
                    pad.left_y = v;
                }
                if let Some(v) = range.rx {
                    pad.right_x = v;
                }
                if let Some(v) = range.ry {
                    pad.right_y = v;
                }
            }
        }
        state
    }

    pub(crate) fn keys_at(&self, wframe: u64) -> Vec<KeyState> {
        if wframe == 0 {
            return Vec::new();
        }
        let mut keys = match self.key_traces.partition_point(|e| e.wframe <= wframe) {
            0 => Vec::new(),
            idx => self.key_traces[idx - 1].keys.clone(),
        };
        for range in &self.ranges {
            if range.from <= wframe && wframe <= range.to {
                for &k in &range.keys {
                    if !keys.contains(&k) {
                        keys.push(k);
                    }
                }
            }
        }
        keys
    }
}

struct Recorder {
    writer: BufWriter<File>,
    started: Instant,
    last_pads: [Option<Option<PadState>>; PADS],
    last_keys: Option<Vec<KeyState>>,
}

impl Recorder {
    fn create(path: &Path, replay_path: Option<&Path>) -> Result<Self, String> {
        victory_gpu::bench::ensure_outside_repo(path)?;
        let same_input = "VICTORY_INPUT_RECORD points to the same file as VICTORY_INPUT_REPLAY";
        if replay_path == Some(path) {
            return Err(same_input.to_owned());
        }
        let replay_file = match replay_path.map(same_file::Handle::from_path).transpose() {
            Ok(handle) => handle,
            Err(err) if err.kind() == ErrorKind::NotFound => None,
            Err(err) => {
                return Err(format!(
                    "cannot check the identity of VICTORY_INPUT_REPLAY: {err}"
                ));
            }
        };
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create parent dir for {}: {e}", path.display()))?;
        }
        let open_error = |err| format!("cannot open {} for input recording: {err}", path.display());
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(replay_file.is_none())
            .open(path)
            .map_err(open_error)?;
        if let Some(replay_file) = replay_file {
            // Compare the open files before truncating so aliases cannot erase the replay.
            let recording_file =
                same_file::Handle::from_file(file.try_clone().map_err(open_error)?)
                    .map_err(open_error)?;
            if recording_file == replay_file {
                return Err(same_input.to_owned());
            }
            file.set_len(0).map_err(|err| {
                format!(
                    "cannot truncate {} for input recording: {err}",
                    path.display()
                )
            })?;
        }
        Ok(Self {
            writer: BufWriter::new(file),
            started: Instant::now(),
            last_pads: [None; PADS],
            last_keys: None,
        })
    }

    fn record_pad(&mut self, wframe: u64, slot: usize, pad: Option<PadState>) {
        if wframe == 0 || slot >= PADS {
            return;
        }
        if self.last_pads[slot].as_ref() == Some(&pad) {
            return;
        }
        self.last_pads[slot] = Some(pad);
        let ms = self.started.elapsed().as_millis() as u64;
        let line = match pad {
            Some(p) => format!(
                "{{\"wframe\":{},\"ms\":{},\"slot\":{},\"connected\":true,\"buttons\":{},\"lt\":{},\"rt\":{},\"lx\":{},\"ly\":{},\"rx\":{},\"ry\":{}}}\n",
                wframe,
                ms,
                slot,
                p.buttons,
                format_f32(p.left_trigger),
                format_f32(p.right_trigger),
                format_f32(p.left_x),
                format_f32(p.left_y),
                format_f32(p.right_x),
                format_f32(p.right_y),
            ),
            None => format!(
                "{{\"wframe\":{wframe},\"ms\":{ms},\"slot\":{slot},\"connected\":false}}\n"
            ),
        };
        let _ = self.writer.write_all(line.as_bytes());
        let _ = self.writer.flush();
    }

    fn record_keys(&mut self, wframe: u64, keys: &[KeyState]) {
        if wframe == 0 {
            return;
        }
        if self.last_keys.as_deref() == Some(keys) {
            return;
        }
        self.last_keys = Some(keys.to_vec());
        let ms = self.started.elapsed().as_millis() as u64;
        let mut line = format!("{{\"wframe\":{wframe},\"ms\":{ms},\"keys\":[");
        for (i, k) in keys.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            use std::fmt::Write;
            let _ = write!(
                line,
                "{{\"vk\":{},\"scan\":{},\"cp\":{},\"dead\":{}}}",
                k.virtual_key, k.scan_code, k.code_point, k.dead_key
            );
        }
        line.push_str("]}\n");
        let _ = self.writer.write_all(line.as_bytes());
        let _ = self.writer.flush();
    }
}

fn format_f32(v: f32) -> String {
    if !v.is_finite() {
        return "0.0".to_owned();
    }
    let s = v.to_string();
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

pub(crate) struct ScriptedSource {
    inner: Box<dyn Source>,
    replay: Option<Script>,
    recorder: Option<Recorder>,
    world_frame: fn() -> u64,
}

impl ScriptedSource {
    fn with_fn(
        inner: Box<dyn Source>,
        replay: Option<Script>,
        recorder: Option<Recorder>,
        world_frame: fn() -> u64,
    ) -> Self {
        Self {
            inner,
            replay,
            recorder,
            world_frame,
        }
    }
}

impl Source for ScriptedSource {
    fn pad(&mut self, slot: usize) -> Option<PadState> {
        let wf = (self.world_frame)();
        let pad = match &self.replay {
            Some(script) => script.pad_at(wf, slot),
            None => self.inner.pad(slot),
        };
        if let Some(rec) = &mut self.recorder {
            rec.record_pad(wf, slot, pad);
        }
        pad
    }

    fn keys(&mut self) -> Vec<KeyState> {
        let wf = (self.world_frame)();
        // While `wf == 0` (menus/loading screens), pass through underlying keys
        // so `VICTORY_KEYS` or keyboard navigation can reach the 3D world; once
        // `wf >= 1`, apply the frame-indexed replay script exclusively.
        let keys = match &self.replay {
            Some(script) if wf > 0 => script.keys_at(wf),
            _ => self.inner.keys(),
        };
        if let Some(rec) = &mut self.recorder {
            rec.record_keys(wf, &keys);
        }
        keys
    }

    #[cfg(any(test, feature = "diagnostics"))]
    fn scripted_keys(&mut self) -> Vec<KeyState> {
        let wf = (self.world_frame)();
        match &self.replay {
            Some(_) if wf > 0 => Vec::new(),
            _ => self.inner.scripted_keys(),
        }
    }

    fn rumble(&mut self, slot: usize, low: u16, high: u16) {
        if self.replay.is_none() {
            self.inner.rumble(slot, low, high);
        }
    }
}

fn parse_replay(
    bytes: &[u8],
    register: impl FnOnce(&[u8]) -> String,
) -> Result<(Script, String), String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("invalid UTF-8: {e}"))?;
    let script = Script::parse(text)?;
    Ok((script, register(bytes)))
}

/// Wraps `source` with input replay and/or recording when `VICTORY_INPUT_REPLAY`
/// or `VICTORY_INPUT_RECORD` is configured.
pub(crate) fn wrap(source: Box<dyn Source>) -> Box<dyn Source> {
    let replay_path = std::env::var_os("VICTORY_INPUT_REPLAY")
        .map(|v| v.to_string_lossy().trim().to_owned())
        .filter(|s| !s.is_empty());
    let record_path = std::env::var_os("VICTORY_INPUT_RECORD")
        .map(|v| v.to_string_lossy().trim().to_owned())
        .filter(|s| !s.is_empty());

    wrap_with_paths(
        source,
        replay_path.as_deref().map(Path::new),
        record_path.as_deref().map(Path::new),
    )
}

fn wrap_with_paths(
    source: Box<dyn Source>,
    replay_path: Option<&Path>,
    record_path: Option<&Path>,
) -> Box<dyn Source> {
    let replay = replay_path.and_then(|path| match std::fs::read(path) {
        Ok(bytes) => match parse_replay(&bytes, victory_gpu::bench::register_input_script) {
            Ok((script, sha)) => {
                tracing::info!(path = %path.display(), sha256 = %sha, "replaying scripted world-frame inputs");
                Some(script)
            }
            Err(err) => {
                tracing::error!(path = %path.display(), error = %err, "failed to parse VICTORY_INPUT_REPLAY");
                None
            }
        },
        Err(err) => {
            tracing::error!(path = %path.display(), error = %err, "failed to read VICTORY_INPUT_REPLAY");
            None
        }
    });

    let recorder = record_path.and_then(|path| match Recorder::create(path, replay_path) {
        Ok(rec) => {
            tracing::info!(path = %path.display(), "recording world-frame inputs");
            Some(rec)
        }
        Err(err) => {
            tracing::error!(path = %path.display(), error = %err, "failed to open VICTORY_INPUT_RECORD");
            None
        }
    });

    if replay.is_none() && recorder.is_none() {
        return source;
    }

    Box::new(ScriptedSource::with_fn(
        source,
        replay,
        recorder,
        victory_gpu::bench::world_frame,
    ))
}

#[derive(Clone, Debug)]
enum JsonVal {
    Bool(bool),
    Num(String),
    Array(Vec<JsonVal>),
    Object(HashMap<String, JsonVal>),
    Null,
    Str(String),
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if b.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, ch: u8) -> Result<(), String> {
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&ch) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", ch as char, self.pos))
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        while let Some(&b) = self.bytes.get(self.pos) {
            self.pos += 1;
            match b {
                b'"' => {
                    return String::from_utf8(out)
                        .map_err(|e| format!("invalid UTF-8 in JSON string: {e}"));
                }
                b'\\' => {
                    let esc = *self
                        .bytes
                        .get(self.pos)
                        .ok_or_else(|| "unterminated escape".to_owned())?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        _ => return Err(format!("unsupported escape \\{}", esc as char)),
                    }
                }
                c => out.push(c),
            }
        }
        Err("unterminated string".to_owned())
    }

    fn parse_number(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(&b) = self.bytes.get(self.pos) {
            if b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(format!("expected number at byte {}", self.pos));
        }
        std::str::from_utf8(&self.bytes[start..self.pos])
            .map(str::to_owned)
            .map_err(|e| e.to_string())
    }

    fn parse_value(&mut self) -> Result<JsonVal, String> {
        self.skip_ws();
        let Some(&b) = self.bytes.get(self.pos) else {
            return Err("unexpected end of JSON".to_owned());
        };
        match b {
            b'{' => Ok(JsonVal::Object(self.parse_object()?)),
            b'[' => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_ws();
                if self.bytes.get(self.pos) == Some(&b']') {
                    self.pos += 1;
                    return Ok(JsonVal::Array(items));
                }
                loop {
                    items.push(self.parse_value()?);
                    self.skip_ws();
                    match self.bytes.get(self.pos) {
                        Some(&b',') => self.pos += 1,
                        Some(&b']') => {
                            self.pos += 1;
                            return Ok(JsonVal::Array(items));
                        }
                        _ => return Err(format!("expected ',' or ']' at byte {}", self.pos)),
                    }
                }
            }
            b'"' => Ok(JsonVal::Str(self.parse_string()?)),
            b't' if self.bytes[self.pos..].starts_with(b"true") => {
                self.pos += 4;
                Ok(JsonVal::Bool(true))
            }
            b'f' if self.bytes[self.pos..].starts_with(b"false") => {
                self.pos += 5;
                Ok(JsonVal::Bool(false))
            }
            b'n' if self.bytes[self.pos..].starts_with(b"null") => {
                self.pos += 4;
                Ok(JsonVal::Null)
            }
            _ => Ok(JsonVal::Num(self.parse_number()?)),
        }
    }

    fn parse_object(&mut self) -> Result<HashMap<String, JsonVal>, String> {
        self.expect(b'{')?;
        let mut map = HashMap::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(map);
        }
        loop {
            let key = self.parse_string()?;
            self.expect(b':')?;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(&b',') => self.pos += 1,
                Some(&b'}') => {
                    self.pos += 1;
                    return Ok(map);
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.pos)),
            }
        }
    }
}

fn parse_json_object(line: &str) -> Result<HashMap<String, JsonVal>, String> {
    let mut parser = JsonParser::new(line);
    let obj = parser.parse_object()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(format!("trailing characters at byte {}", parser.pos));
    }
    Ok(obj)
}

fn obj_u64(obj: &HashMap<String, JsonVal>, key: &str) -> Result<Option<u64>, String> {
    match obj.get(key) {
        None | Some(JsonVal::Null) => Ok(None),
        Some(JsonVal::Num(s)) => s
            .parse::<u64>()
            .map(Some)
            .map_err(|e| format!("invalid integer `{key}` ({s}): {e}")),
        _ => Err(format!("expected integer for `{key}`")),
    }
}

fn obj_f32(obj: &HashMap<String, JsonVal>, key: &str) -> Result<Option<f32>, String> {
    match obj.get(key) {
        None | Some(JsonVal::Null) => Ok(None),
        Some(JsonVal::Num(s)) => s
            .parse::<f32>()
            .map(Some)
            .map_err(|e| format!("invalid float `{key}` ({s}): {e}")),
        _ => Err(format!("expected float for `{key}`")),
    }
}

fn obj_bool(obj: &HashMap<String, JsonVal>, key: &str) -> Result<Option<bool>, String> {
    match obj.get(key) {
        None | Some(JsonVal::Null) => Ok(None),
        Some(JsonVal::Bool(b)) => Ok(Some(*b)),
        _ => Err(format!("expected bool for `{key}`")),
    }
}

fn obj_keys(obj: &HashMap<String, JsonVal>, key: &str) -> Result<Vec<KeyState>, String> {
    match obj.get(key) {
        None | Some(JsonVal::Null) => Ok(Vec::new()),
        Some(v) => parse_keys_value(v),
    }
}

fn parse_keys_value(val: &JsonVal) -> Result<Vec<KeyState>, String> {
    let JsonVal::Array(arr) = val else {
        return Err("expected array for `keys`".to_owned());
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        match item {
            JsonVal::Object(map) => {
                let vk = obj_u64(map, "vk")?
                    .or(obj_u64(map, "virtual_key")?)
                    .ok_or_else(|| "key object missing `vk`".to_owned())?
                    as u8;
                let scan = obj_u64(map, "scan")?
                    .or(obj_u64(map, "scan_code")?)
                    .unwrap_or(0) as u32;
                let cp = obj_u64(map, "cp")?
                    .or(obj_u64(map, "code_point")?)
                    .unwrap_or(0) as u32;
                let dead = obj_bool(map, "dead")?
                    .or(obj_bool(map, "dead_key")?)
                    .unwrap_or(false);
                out.push(KeyState {
                    scan_code: scan,
                    code_point: cp,
                    virtual_key: vk,
                    dead_key: dead,
                });
            }
            JsonVal::Num(s) => {
                let vk = s
                    .parse::<u8>()
                    .map_err(|e| format!("invalid key code {s}: {e}"))?;
                out.push(KeyState {
                    virtual_key: vk,
                    ..Default::default()
                });
            }
            JsonVal::Str(s) if s.len() == 1 => {
                out.push(KeyState {
                    virtual_key: s.as_bytes()[0].to_ascii_uppercase(),
                    ..Default::default()
                });
            }
            _ => return Err("unsupported entry inside `keys`".to_owned()),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn rejected_replays_are_not_registered_for_benchmarks() {
        for bytes in [b"not a replay".as_slice(), &[0xff]] {
            assert!(parse_replay(bytes, |_| panic!("registered a rejected replay")).is_err());
        }

        let valid = b"{\"from\":1,\"to\":2,\"buttons\":16}\n";
        let (script, digest) = parse_replay(valid, |bytes| {
            assert_eq!(bytes, valid);
            "digest".to_owned()
        })
        .unwrap();
        assert_eq!(script.pad_at(1, 0).unwrap().buttons, 16);
        assert_eq!(digest, "digest");
    }

    #[derive(Default)]
    struct MockHost {
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        scripted: Vec<KeyState>,
    }

    struct MockSource(Arc<Mutex<MockHost>>);

    impl Source for MockSource {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            self.0.lock().unwrap().pads[slot]
        }
        fn keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().keys.clone()
        }
        fn scripted_keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().scripted.clone()
        }
        fn rumble(&mut self, _: usize, _: u16, _: u16) {}
    }

    #[test]
    fn recording_does_not_overwrite_the_replay_input() {
        let dir = tempfile::tempdir().unwrap();
        let replay = dir.path().join("replay.jsonl");
        let trace = br#"{"from":1,"to":2,"buttons":16}"#;
        std::fs::write(&replay, trace).unwrap();
        let source = Box::new(MockSource(Arc::default()));
        drop(wrap_with_paths(source, Some(&replay), Some(&replay)));
        assert_eq!(std::fs::read(&replay).unwrap(), trace);
    }

    #[test]
    fn recording_does_not_overwrite_a_hard_link_to_a_rejected_replay() {
        let dir = tempfile::tempdir().unwrap();
        let replay = dir.path().join("replay.jsonl");
        let record = dir.path().join("record.jsonl");
        let trace = b"not a valid replay";
        std::fs::write(&replay, trace).unwrap();
        std::fs::hard_link(&replay, &record).unwrap();
        let source = Box::new(MockSource(Arc::default()));
        drop(wrap_with_paths(source, Some(&replay), Some(&record)));
        assert_eq!(std::fs::read(&replay).unwrap(), trace);
    }

    #[test]
    fn recording_can_overwrite_a_separate_output_file_while_replaying() {
        let dir = tempfile::tempdir().unwrap();
        let replay = dir.path().join("replay.jsonl");
        let record = dir.path().join("record.jsonl");
        let trace = br#"{"from":1,"to":2,"buttons":16}"#;
        std::fs::write(&replay, trace).unwrap();
        std::fs::write(&record, trace).unwrap();
        let source = Box::new(MockSource(Arc::default()));
        drop(wrap_with_paths(source, Some(&replay), Some(&record)));
        assert_eq!(std::fs::read(&replay).unwrap(), trace);
        assert!(std::fs::read(&record).unwrap().is_empty());
    }

    static TEST_WFRAME: AtomicU64 = AtomicU64::new(0);
    fn test_world_frame() -> u64 {
        TEST_WFRAME.load(Ordering::Relaxed)
    }

    #[test]
    fn record_then_replay_roundtrips_pad_state_and_keys_bit_exact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let host = Arc::new(Mutex::new(MockHost::default()));

        let rec = Recorder::create(&path, None).unwrap();
        let mut src = ScriptedSource::with_fn(
            Box::new(MockSource(Arc::clone(&host))),
            None,
            Some(rec),
            test_world_frame,
        );

        let pad_a = PadState {
            buttons: 0x0405,
            left_trigger: 0.125,
            right_trigger: 0.8125,
            left_x: -0.33333334,
            left_y: 0.9999695,
            right_x: 0.25,
            right_y: -0.75,
        };
        let pad_b = PadState {
            buttons: 0,
            left_y: -0.5,
            ..Default::default()
        };
        let key_w = KeyState {
            scan_code: 0x11,
            code_point: b'w' as u32,
            virtual_key: b'W',
            dead_key: false,
        };

        // Loading screen (wframe == 0) is not recorded.
        TEST_WFRAME.store(0, Ordering::Relaxed);
        host.lock().unwrap().pads[0] = Some(pad_a);
        assert_eq!(src.pad(0), Some(pad_a));

        TEST_WFRAME.store(1, Ordering::Relaxed);
        assert_eq!(src.pad(0), Some(pad_a));

        TEST_WFRAME.store(5, Ordering::Relaxed);
        host.lock().unwrap().keys = vec![key_w];
        assert_eq!(src.keys(), vec![key_w]);

        TEST_WFRAME.store(10, Ordering::Relaxed);
        host.lock().unwrap().pads[0] = Some(pad_b);
        host.lock().unwrap().keys.clear();
        assert_eq!(src.pad(0), Some(pad_b));
        assert!(src.keys().is_empty());
        drop(src);

        let raw = std::fs::read_to_string(&path).unwrap();
        let script = Script::parse(&raw).unwrap();

        // Loading screen holds neutral pad 0 and empty script keys.
        assert_eq!(script.pad_at(0, 0), Some(PadState::default()));
        assert!(script.keys_at(0).is_empty());

        let got_1 = script.pad_at(1, 0).unwrap();
        assert_eq!(got_1.buttons, pad_a.buttons);
        assert_eq!(got_1.left_x.to_bits(), pad_a.left_x.to_bits());
        assert_eq!(got_1.left_y.to_bits(), pad_a.left_y.to_bits());
        assert_eq!(got_1.right_trigger.to_bits(), pad_a.right_trigger.to_bits());
        assert_eq!(script.keys_at(5), vec![key_w]);

        let got_10 = script.pad_at(10, 0).unwrap();
        assert_eq!(got_10.buttons, pad_b.buttons);
        assert_eq!(got_10.left_y.to_bits(), pad_b.left_y.to_bits());
        assert!(script.keys_at(10).is_empty());
    }

    #[test]
    fn declarative_range_scripts_wait_on_loading_screen_and_apply_in_world() {
        let text = concat!(
            "# move forward for 120 world frames\n",
            "{\"from\":0,\"to\":120,\"ly\":1.0,\"rt\":0.8}\n",
            "{\"from\":50,\"to\":60,\"buttons\":4,\"keys\":[87]}\n",
        );
        let script = Script::parse(text).unwrap();

        // Loading screen (`wframe == 0`) stays neutral.
        assert_eq!(script.pad_at(0, 0), Some(PadState::default()));
        assert!(script.keys_at(0).is_empty());

        let p1 = script.pad_at(1, 0).unwrap();
        assert_eq!(p1.left_y, 1.0);
        assert_eq!(p1.right_trigger, 0.8);
        assert_eq!(p1.buttons, 0);

        let p55 = script.pad_at(55, 0).unwrap();
        assert_eq!(p55.left_y, 1.0);
        assert_eq!(p55.right_trigger, 0.8);
        assert_eq!(p55.buttons, 4);
        assert_eq!(script.keys_at(55).len(), 1);
        assert_eq!(script.keys_at(55)[0].virtual_key, 87);

        // After `to`, returns to rest.
        assert_eq!(script.pad_at(121, 0), Some(PadState::default()));
        assert!(script.keys_at(121).is_empty());

        // Recording rejects paths inside any `.victory-root` repository tree.
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("input-trace.jsonl");
        assert!(Recorder::create(&inside, None).is_err());

        // Multi-byte UTF-8 characters inside JSON strings decode without Latin-1 corruption.
        let utf8_script =
            Script::parse("{\"from\":1,\"to\":2,\"note\":\"тест — ✓\",\"ly\":0.5}\n").unwrap();
        assert_eq!(utf8_script.pad_at(1, 0).unwrap().left_y, 0.5);
    }

    /// While `world_frame() == 0` in replay mode, gamepad stays neutral so
    /// loading never drifts, while underlying `keys()` and `scripted_keys()`
    /// pass through so `VICTORY_KEYS` menu navigation still drives loading.
    /// Once `world_frame() >= 1`, the replay script takes over exclusively.
    #[test]
    fn replay_passes_through_loading_keys_at_world_frame_zero_then_replays_in_world() {
        static REPLAY_WF: AtomicU64 = AtomicU64::new(0);
        fn replay_wf() -> u64 {
            REPLAY_WF.load(Ordering::Relaxed)
        }

        let enter = KeyState {
            virtual_key: 0x0D,
            ..Default::default()
        };
        let script_key_w = KeyState {
            virtual_key: b'W',
            ..Default::default()
        };
        let host = Arc::new(Mutex::new(MockHost {
            pads: [
                Some(PadState {
                    buttons: 0xFF,
                    left_y: -1.0,
                    ..Default::default()
                }),
                None,
                None,
                None,
            ],
            keys: vec![enter],
            scripted: vec![enter],
        }));

        let script = Script::parse("{\"from\":1,\"to\":10,\"ly\":1.0,\"keys\":[87]}\n").unwrap();
        let mut src = ScriptedSource::with_fn(
            Box::new(MockSource(Arc::clone(&host))),
            Some(script),
            None,
            replay_wf,
        );

        // Menus/loading (`wframe == 0`): pad held neutral, keys & scripted_keys pass through.
        REPLAY_WF.store(0, Ordering::Relaxed);
        assert_eq!(src.pad(0), Some(PadState::default()));
        assert_eq!(src.pad(1), None);
        assert_eq!(src.keys(), vec![enter]);
        assert_eq!(src.scripted_keys(), vec![enter]);

        // World frame 1: replay script drives pad and keys; underlying menu keys suppressed.
        REPLAY_WF.store(1, Ordering::Relaxed);
        let p1 = src.pad(0).unwrap();
        assert_eq!(p1.left_y, 1.0);
        assert_eq!(p1.buttons, 0);
        assert_eq!(src.keys(), vec![script_key_w]);
        assert!(src.scripted_keys().is_empty());

        // Past range end: replay returns neutral pad and empty keys (still not leaking host keys).
        REPLAY_WF.store(11, Ordering::Relaxed);
        assert_eq!(src.pad(0), Some(PadState::default()));
        assert!(src.keys().is_empty());
        assert!(src.scripted_keys().is_empty());
    }

    #[test]
    fn variable_fps_replay_produces_identical_inputs_per_world_frame() {
        let script = Script::parse(concat!(
            "{\"from\":1,\"to\":5,\"ly\":1.0}\n",
            "{\"from\":6,\"to\":10,\"rx\":0.5,\"rt\":0.75}\n",
        ))
        .unwrap();

        let simulate = |loading_polls: usize, polls_per_world_frame: usize| {
            let wf = Arc::new(AtomicU64::new(0));
            let mut out = Vec::new();
            for _ in 0..loading_polls {
                out.push((0, script.pad_at(wf.load(Ordering::Relaxed), 0)));
            }
            for frame in 1..=11u64 {
                wf.store(frame, Ordering::Relaxed);
                for _ in 0..polls_per_world_frame {
                    out.push((frame, script.pad_at(frame, 0)));
                }
            }
            out
        };

        let slow_20fps = simulate(15, 1);
        let fast_120fps = simulate(3, 4);
        for frame in 1..=11u64 {
            let a: Vec<_> = slow_20fps
                .iter()
                .filter(|(f, _)| *f == frame)
                .map(|(_, p)| *p)
                .collect();
            let b: Vec<_> = fast_120fps
                .iter()
                .filter(|(f, _)| *f == frame)
                .map(|(_, p)| *p)
                .collect();
            assert!(!a.is_empty() && !b.is_empty());
            assert!(a.iter().all(|p| *p == a[0]));
            assert!(b.iter().all(|p| *p == a[0]));
        }
    }
}
