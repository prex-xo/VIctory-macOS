//! A silent playback endpoint for hosts with no active render device.
//!
//! The title plays its own mix through Windows audio: it makes an
//! `IMMDeviceEnumerator` with `CoCreateInstance`, takes the default render
//! endpoint and activates an `ISpatialAudioClient` on it. With no active
//! endpoint its output setup never allocates buffers and returns holding a
//! lock, so the audio thread never drains the title's command streams and
//! the title's audio-thread assert fires. A console always has an output.
//!
//! `CoCreateInstance` is ours. Everything but the device enumerator goes to
//! the host unchanged, and so does an enumerator that reports an active
//! render endpoint. Otherwise the enumerator is wrapped: render flow shows
//! one silent device, capture flow is the host's. The device's spatial
//! client takes the title's stream, signals its event once per period and
//! discards what the title writes.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{
    E_FAIL, E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL, E_POINTER, HANDLE,
};
use windows::Win32::Media::Audio::{
    AUDCLNT_E_UNSUPPORTED_FORMAT, AudioObjectType, AudioObjectType_BackLeft,
    AudioObjectType_BackRight, AudioObjectType_Dynamic, AudioObjectType_FrontCenter,
    AudioObjectType_FrontLeft, AudioObjectType_FrontRight, AudioObjectType_LowFrequency,
    AudioObjectType_SideLeft, AudioObjectType_SideRight, DEVICE_STATE, DEVICE_STATE_ACTIVE,
    EDataFlow, ERole, IAudioFormatEnumerator, IAudioFormatEnumerator_Impl, IMMDevice,
    IMMDevice_Impl, IMMDeviceCollection, IMMDeviceCollection_Impl, IMMDeviceEnumerator,
    IMMDeviceEnumerator_Impl, IMMNotificationClient, ISpatialAudioClient, ISpatialAudioClient_Impl,
    ISpatialAudioObject, ISpatialAudioObject_Impl, ISpatialAudioObjectBase_Impl,
    ISpatialAudioObjectRenderStream, ISpatialAudioObjectRenderStream_Impl,
    ISpatialAudioObjectRenderStreamBase_Impl, MMDeviceEnumerator, SPTLAUDCLNT_E_NO_MORE_OBJECTS,
    SpatialAudioObjectRenderStreamActivationParams, WAVEFORMATEX, eRender,
};
use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
use windows::Win32::System::Com::{CLSCTX, CoTaskMemAlloc, STGM};
use windows::Win32::System::Threading::SetEvent;
use windows::Win32::UI::Shell::PropertiesSystem::IPropertyStore;
use windows::core::{
    BOOL, GUID, HRESULT, IUnknown, Interface, PCWSTR, PWSTR, Ref, Result, implement,
};

use super::super::files::host;
use crate::{HleModule, hle_export};

/// What the title is offered: 48 kHz mono float objects, 10 ms at a time.
const RATE: u32 = 48_000;
const FRAMES: u32 = RATE / 100;
const FORMAT: WAVEFORMATEX = WAVEFORMATEX {
    wFormatTag: 3, // WAVE_FORMAT_IEEE_FLOAT
    nChannels: 1,
    nSamplesPerSec: RATE,
    nAvgBytesPerSec: RATE * 4,
    nBlockAlign: 4,
    wBitsPerSample: 32,
    cbSize: 0,
};
/// A 7.1 bed.
const STATIC_OBJECTS: i32 = AudioObjectType_FrontLeft.0
    | AudioObjectType_FrontRight.0
    | AudioObjectType_FrontCenter.0
    | AudioObjectType_LowFrequency.0
    | AudioObjectType_SideLeft.0
    | AudioObjectType_SideRight.0
    | AudioObjectType_BackLeft.0
    | AudioObjectType_BackRight.0;
const DYNAMIC_OBJECTS: u32 = 111;

/// The silent device's id, stable between runs.
const DEVICE_ID: &str = "{0.0.0.00000000}.{76696374-6f72-7953-696c-656e74000000}";

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn is_active_render(flow: EDataFlow, mask: DEVICE_STATE) -> bool {
    flow == eRender && mask.0 & DEVICE_STATE_ACTIVE.0 != 0
}

// The stream.

/// Tells the timer thread whether the title has the stream started.
struct Clock {
    /// `(running, quit)`.
    state: Mutex<(bool, bool)>,
    changed: Condvar,
}

/// Signals `event` once per period while the clock runs. Periods are kept
/// on a fixed schedule, so a late wake-up is made up for, not added to.
fn tick(clock: &Clock, event: usize, period: Duration) {
    let mut state = lock(&clock.state);
    let mut next = Instant::now() + period;
    loop {
        if state.1 {
            return;
        }
        if !state.0 {
            state = clock.changed.wait(state).unwrap_or_else(|e| e.into_inner());
            next = Instant::now() + period;
            continue;
        }
        let now = Instant::now();
        if now >= next {
            // SAFETY: the title's event handle, valid while it keeps the stream.
            let _ = unsafe { SetEvent(HANDLE(event as *mut c_void)) };
            next += period;
            // After a long stall (a suspended host) skip the missed
            // periods; the title only needs the event to keep coming.
            if next < now {
                next = now + period;
            }
        } else {
            state = clock
                .changed
                .wait_timeout(state, next - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }
}

/// What the title asked for in `ActivateSpatialAudioStream`.
struct Request {
    event: usize,
    max_dynamic: u32,
    static_mask: i32,
}

#[implement(ISpatialAudioObjectRenderStream)]
struct SilentStream {
    clock: Arc<Clock>,
    thread: Option<JoinHandle<()>>,
    request: Request,
    /// Dynamic objects the title holds.
    dynamic: Arc<AtomicU32>,
}

impl SilentStream {
    fn new(request: Request) -> Self {
        let clock = Arc::new(Clock {
            state: Mutex::new((false, false)),
            changed: Condvar::new(),
        });
        let thread = std::thread::Builder::new()
            .name("victory silent audio".into())
            .spawn({
                let clock = clock.clone();
                let event = request.event;
                move || tick(&clock, event, Duration::from_millis(10))
            })
            .ok();
        SilentStream {
            clock,
            thread,
            request,
            dynamic: Arc::default(),
        }
    }

    fn run(&self, running: bool) -> Result<()> {
        lock(&self.clock.state).0 = running;
        self.clock.changed.notify_all();
        Ok(())
    }
}

impl Drop for SilentStream {
    fn drop(&mut self) {
        lock(&self.clock.state).1 = true;
        self.clock.changed.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl ISpatialAudioObjectRenderStreamBase_Impl for SilentStream_Impl {
    fn GetAvailableDynamicObjectCount(&self) -> Result<u32> {
        let held = self.dynamic.load(Ordering::Relaxed);
        Ok(self.request.max_dynamic.saturating_sub(held))
    }
    fn GetService(&self, _: *const GUID, _: *mut *mut c_void) -> Result<()> {
        Err(E_NOINTERFACE.into())
    }
    fn Start(&self) -> Result<()> {
        self.run(true)
    }
    fn Stop(&self) -> Result<()> {
        self.run(false)
    }
    fn Reset(&self) -> Result<()> {
        Ok(())
    }
    fn BeginUpdatingAudioObjects(&self, available: *mut u32, frames: *mut u32) -> Result<()> {
        if available.is_null() || frames.is_null() {
            return Err(E_POINTER.into());
        }
        // SAFETY: both are non-null out parameters.
        unsafe {
            *available = self.GetAvailableDynamicObjectCount()?;
            *frames = FRAMES;
        }
        Ok(())
    }
    fn EndUpdatingAudioObjects(&self) -> Result<()> {
        Ok(())
    }
}

impl ISpatialAudioObjectRenderStream_Impl for SilentStream_Impl {
    fn ActivateSpatialAudioObject(&self, kind: AudioObjectType) -> Result<ISpatialAudioObject> {
        let dynamic = kind == AudioObjectType_Dynamic;
        if dynamic {
            let taken = self.dynamic.fetch_add(1, Ordering::Relaxed);
            if taken >= self.request.max_dynamic {
                self.dynamic.fetch_sub(1, Ordering::Relaxed);
                return Err(SPTLAUDCLNT_E_NO_MORE_OBJECTS.into());
            }
        } else if kind.0 & self.request.static_mask == 0 {
            return Err(E_INVALIDARG.into());
        }
        Ok(SilentObject {
            kind,
            buffer: Mutex::new(vec![0; FRAMES as usize * 4]),
            held: dynamic.then(|| self.dynamic.clone()),
        }
        .into())
    }
}

/// An audio object: a buffer for the title to write and nothing behind it.
#[implement(ISpatialAudioObject)]
struct SilentObject {
    kind: AudioObjectType,
    buffer: Mutex<Vec<u8>>,
    held: Option<Arc<AtomicU32>>,
}

impl Drop for SilentObject {
    fn drop(&mut self) {
        if let Some(held) = &self.held {
            held.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl ISpatialAudioObjectBase_Impl for SilentObject_Impl {
    fn GetBuffer(&self, buffer: *mut *mut u8, length: *mut u32) -> Result<()> {
        if buffer.is_null() || length.is_null() {
            return Err(E_POINTER.into());
        }
        let mut samples = lock(&self.buffer);
        // SAFETY: non-null out parameters. The buffer is never resized, so
        // the pointer stays valid for as long as the object lives.
        unsafe {
            *buffer = samples.as_mut_ptr();
            *length = samples.len() as u32;
        }
        Ok(())
    }
    fn SetEndOfStream(&self, _: u32) -> Result<()> {
        Ok(())
    }
    fn IsActive(&self) -> Result<BOOL> {
        Ok(true.into())
    }
    fn GetAudioObjectType(&self) -> Result<AudioObjectType> {
        Ok(self.kind)
    }
}

impl ISpatialAudioObject_Impl for SilentObject_Impl {
    fn SetPosition(&self, _: f32, _: f32, _: f32) -> Result<()> {
        Ok(())
    }
    fn SetVolume(&self, _: f32) -> Result<()> {
        Ok(())
    }
}

// The client.

/// The one format the endpoint lists.
#[implement(IAudioFormatEnumerator)]
struct Formats(Box<WAVEFORMATEX>);

impl IAudioFormatEnumerator_Impl for Formats_Impl {
    fn GetCount(&self) -> Result<u32> {
        Ok(1)
    }
    fn GetFormat(&self, index: u32) -> Result<*mut WAVEFORMATEX> {
        match index {
            0 => Ok(&raw const *self.0 as *mut WAVEFORMATEX),
            _ => Err(E_INVALIDARG.into()),
        }
    }
}

/// Float samples at the endpoint's rate; the title's formats are 48 kHz.
fn check_format(format: *const WAVEFORMATEX) -> Result<()> {
    if format.is_null() {
        return Err(E_POINTER.into());
    }
    // SAFETY: non-null, and a WAVEFORMATEX is at least this big.
    let format = unsafe { format.read_unaligned() };
    let (tag, rate, bits) = (
        format.wFormatTag,
        format.nSamplesPerSec,
        format.wBitsPerSample,
    );
    // WAVE_FORMAT_IEEE_FLOAT, or WAVE_FORMAT_EXTENSIBLE (a float one in
    // practice: the sub-format follows the header).
    if (tag == 3 || tag == 0xFFFE) && rate == RATE && bits == 32 {
        Ok(())
    } else {
        Err(AUDCLNT_E_UNSUPPORTED_FORMAT.into())
    }
}

/// The title's stream request, out of the blob in the activation parameters.
fn request(params: *const PROPVARIANT) -> Result<Request> {
    if params.is_null() {
        return Err(E_POINTER.into());
    }
    // SAFETY: the title passes a VT_BLOB holding the activation parameters.
    let blob = unsafe { (&*params).Anonymous.Anonymous.Anonymous.blob };
    if (blob.cbSize as usize) < size_of::<SpatialAudioObjectRenderStreamActivationParams>() {
        return Err(E_INVALIDARG.into());
    }
    // SAFETY: the blob is at least as big as the (packed) structure.
    let p = unsafe {
        blob.pBlobData
            .cast::<SpatialAudioObjectRenderStreamActivationParams>()
            .read_unaligned()
    };
    check_format(p.ObjectFormat)?;
    Ok(Request {
        event: p.EventHandle.0 as usize,
        max_dynamic: p.MaxDynamicObjectCount.min(DYNAMIC_OBJECTS),
        static_mask: p.StaticObjectTypeMask.0,
    })
}

#[implement(ISpatialAudioClient)]
struct SilentClient;

impl ISpatialAudioClient_Impl for SilentClient_Impl {
    fn GetStaticObjectPosition(
        &self,
        _: AudioObjectType,
        x: *mut f32,
        y: *mut f32,
        z: *mut f32,
    ) -> Result<()> {
        if x.is_null() || y.is_null() || z.is_null() {
            return Err(E_POINTER.into());
        }
        // SAFETY: non-null out parameters.
        unsafe {
            (*x, *y, *z) = (0.0, 0.0, 0.0);
        }
        Ok(())
    }
    fn GetNativeStaticObjectTypeMask(&self) -> Result<AudioObjectType> {
        Ok(AudioObjectType(STATIC_OBJECTS))
    }
    fn GetMaxDynamicObjectCount(&self) -> Result<u32> {
        Ok(DYNAMIC_OBJECTS)
    }
    fn GetSupportedAudioObjectFormatEnumerator(&self) -> Result<IAudioFormatEnumerator> {
        Ok(Formats(Box::new(FORMAT)).into())
    }
    fn GetMaxFrameCount(&self, format: *const WAVEFORMATEX) -> Result<u32> {
        check_format(format)?;
        Ok(FRAMES)
    }
    fn IsAudioObjectFormatSupported(&self, format: *const WAVEFORMATEX) -> Result<()> {
        check_format(format)
    }
    fn IsSpatialAudioStreamAvailable(
        &self,
        stream: *const GUID,
        _: *const PROPVARIANT,
    ) -> Result<()> {
        // SAFETY: the title passes a GUID.
        match unsafe { stream.as_ref() } {
            Some(&ISpatialAudioObjectRenderStream::IID) => Ok(()),
            _ => Err(E_INVALIDARG.into()),
        }
    }
    fn ActivateSpatialAudioStream(
        &self,
        params: *const PROPVARIANT,
        iid: *const GUID,
        stream: *mut *mut c_void,
    ) -> Result<()> {
        if iid.is_null() || stream.is_null() {
            return Err(E_POINTER.into());
        }
        let stream_object: IUnknown =
            ISpatialAudioObjectRenderStream::from(SilentStream::new(request(params)?)).into();
        // SAFETY: non-null out parameters.
        unsafe { stream_object.query(&*iid, stream).ok() }
    }
}

// The device.

#[implement(IMMDevice)]
struct SilentDevice;

impl IMMDevice_Impl for SilentDevice_Impl {
    fn Activate(
        &self,
        iid: *const GUID,
        _: CLSCTX,
        _: *const PROPVARIANT,
        out: *mut *mut c_void,
    ) -> Result<()> {
        if iid.is_null() || out.is_null() {
            return Err(E_POINTER.into());
        }
        // SAFETY: non-null, so the title's GUID and out pointer.
        let (iid, out) = unsafe { (&*iid, out) };
        if *iid != ISpatialAudioClient::IID {
            return Err(E_NOINTERFACE.into());
        }
        let client: ISpatialAudioClient = SilentClient.into();
        // SAFETY: a valid out pointer.
        unsafe { *out = client.into_raw() };
        Ok(())
    }
    fn OpenPropertyStore(&self, _: STGM) -> Result<IPropertyStore> {
        Err(E_NOTIMPL.into())
    }
    fn GetId(&self) -> Result<PWSTR> {
        let id: Vec<u16> = DEVICE_ID.encode_utf16().chain([0]).collect();
        // SAFETY: the caller frees it with CoTaskMemFree, as COM requires.
        unsafe {
            let p = CoTaskMemAlloc(id.len() * 2).cast::<u16>();
            if p.is_null() {
                return Err(E_FAIL.into());
            }
            p.copy_from_nonoverlapping(id.as_ptr(), id.len());
            Ok(PWSTR(p))
        }
    }
    fn GetState(&self) -> Result<DEVICE_STATE> {
        Ok(DEVICE_STATE_ACTIVE)
    }
}

#[implement(IMMDeviceCollection)]
struct Devices(Vec<IMMDevice>);

impl IMMDeviceCollection_Impl for Devices_Impl {
    fn GetCount(&self) -> Result<u32> {
        Ok(self.0.len() as u32)
    }
    fn Item(&self, index: u32) -> Result<IMMDevice> {
        self.0
            .get(index as usize)
            .cloned()
            .ok_or_else(|| E_INVALIDARG.into())
    }
}

// The enumerator.

/// The host's enumerator with the silent device as its render endpoint.
#[implement(IMMDeviceEnumerator)]
struct SilentEnumerator {
    host: IMMDeviceEnumerator,
    device: IMMDevice,
}

impl IMMDeviceEnumerator_Impl for SilentEnumerator_Impl {
    fn EnumAudioEndpoints(
        &self,
        flow: EDataFlow,
        mask: DEVICE_STATE,
    ) -> Result<IMMDeviceCollection> {
        if flow != eRender {
            // SAFETY: the title's arguments, passed on to the host's enumerator.
            return unsafe { self.host.EnumAudioEndpoints(flow, mask) };
        }
        let devices = if is_active_render(flow, mask) {
            vec![self.device.clone()]
        } else {
            Vec::new()
        };
        Ok(Devices(devices).into())
    }
    fn GetDefaultAudioEndpoint(&self, flow: EDataFlow, role: ERole) -> Result<IMMDevice> {
        if flow == eRender {
            Ok(self.device.clone())
        } else {
            // SAFETY: the title's arguments, passed on to the host's enumerator.
            unsafe { self.host.GetDefaultAudioEndpoint(flow, role) }
        }
    }
    fn GetDevice(&self, id: &PCWSTR) -> Result<IMMDevice> {
        // SAFETY: the title passes a NUL-terminated string.
        if unsafe { id.to_string() }.is_ok_and(|id| id == DEVICE_ID) {
            Ok(self.device.clone())
        } else {
            // SAFETY: the title's arguments, passed on to the host's enumerator.
            unsafe { self.host.GetDevice(*id) }
        }
    }
    fn RegisterEndpointNotificationCallback(
        &self,
        client: Ref<IMMNotificationClient>,
    ) -> Result<()> {
        // SAFETY: the title's callback, registered with the host's enumerator.
        unsafe {
            self.host
                .RegisterEndpointNotificationCallback(client.as_ref())
        }
    }
    fn UnregisterEndpointNotificationCallback(
        &self,
        client: Ref<IMMNotificationClient>,
    ) -> Result<()> {
        // SAFETY: the title's callback, registered with the host's enumerator.
        unsafe {
            self.host
                .UnregisterEndpointNotificationCallback(client.as_ref())
        }
    }
}

/// The enumerator the title gets for the host's `native`: the same one when
/// Windows has an active render endpoint, else one with a silent endpoint.
fn route(native: IMMDeviceEnumerator, mute: bool) -> IMMDeviceEnumerator {
    // SAFETY: plain COM calls on a live enumerator. An enumerator that fails
    // to count is the host's problem: the title gets it as it is.
    let active = unsafe {
        native
            .EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE)
            .and_then(|devices| devices.GetCount())
    };
    if !mute && active != Ok(0) {
        return native;
    }
    tracing::info!(mute, "title uses a silent audio endpoint");
    SilentEnumerator {
        host: native,
        device: SilentDevice.into(),
    }
    .into()
}

/// `CoCreateInstance`, with the device enumerator routed by `route`.
extern "system" fn CoCreateInstance(
    clsid: *const GUID,
    outer: *mut c_void,
    context: CLSCTX,
    iid: *const GUID,
    out: *mut *mut c_void,
) -> HRESULT {
    let hr = host!("ole32.dll", "CoCreateInstance": extern "system" fn(*const GUID, *mut c_void, CLSCTX, *const GUID, *mut *mut c_void) -> HRESULT)(
        clsid, outer, context, iid, out,
    );
    // SAFETY: the title's arguments, which the host call has just used.
    let wanted = unsafe {
        hr.is_ok()
            && outer.is_null()
            && clsid.as_ref() == Some(&MMDeviceEnumerator)
            && iid.as_ref() == Some(&IMMDeviceEnumerator::IID)
    };
    if wanted {
        // SAFETY: the host stored an owned IMMDeviceEnumerator.
        let native = unsafe { IMMDeviceEnumerator::from_raw(*out) };
        let mute = std::env::var("VICTORY_SILENT_AUDIO").as_deref() == Ok("1");
        let routed = route(native, mute);
        // SAFETY: as above.
        unsafe { *out = routed.into_raw() };
    }
    hr
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "audio-silent",
        dlls: &["ole32.dll", "api-ms-win-core-com-l1-*"],
        exports: vec![hle_export!(CoCreateInstance)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    /// A host enumerator with `n` active render endpoints and no others.
    #[implement(IMMDeviceEnumerator)]
    struct Fake(u32);

    impl IMMDeviceEnumerator_Impl for Fake_Impl {
        fn EnumAudioEndpoints(&self, _: EDataFlow, _: DEVICE_STATE) -> Result<IMMDeviceCollection> {
            Ok(Devices((0..self.0).map(|_| SilentDevice.into()).collect()).into())
        }
        fn GetDefaultAudioEndpoint(&self, _: EDataFlow, _: ERole) -> Result<IMMDevice> {
            Err(E_NOTIMPL.into())
        }
        fn GetDevice(&self, _: &PCWSTR) -> Result<IMMDevice> {
            Err(E_NOTIMPL.into())
        }
        fn RegisterEndpointNotificationCallback(
            &self,
            _: Ref<IMMNotificationClient>,
        ) -> Result<()> {
            Ok(())
        }
        fn UnregisterEndpointNotificationCallback(
            &self,
            _: Ref<IMMNotificationClient>,
        ) -> Result<()> {
            Ok(())
        }
    }

    fn device_id(device: &IMMDevice) -> String {
        // SAFETY: GetId's string is ours and freed by us.
        unsafe { device.GetId().unwrap().to_string().unwrap() }
    }

    #[test]
    fn an_explicit_mute_routes_playback_to_the_silent_device() {
        let native: IMMDeviceEnumerator = Fake(1).into();
        let routed = route(native, true);
        // SAFETY: the enumerator owns a live playback device.
        let device = unsafe { routed.GetDefaultAudioEndpoint(eRender, ERole(0)).unwrap() };
        assert_eq!(device_id(&device), DEVICE_ID);
    }

    #[test]
    fn a_host_with_an_output_keeps_its_enumerator() {
        let native: IMMDeviceEnumerator = Fake(1).into();
        let routed = route(native.clone(), false);
        assert_eq!(routed.as_raw(), native.as_raw());
    }

    #[test]
    fn a_host_without_one_gets_the_silent_endpoint() {
        let native: IMMDeviceEnumerator = Fake(0).into();
        let routed = route(native.clone(), false);
        assert_ne!(routed.as_raw(), native.as_raw());
        // SAFETY: COM calls on our own objects.
        unsafe {
            let default = routed.GetDefaultAudioEndpoint(eRender, ERole(0)).unwrap();
            assert_eq!(device_id(&default), DEVICE_ID);
            assert_eq!(default.GetState().unwrap(), DEVICE_STATE_ACTIVE);
            let active = routed
                .EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE)
                .unwrap();
            assert_eq!(active.GetCount().unwrap(), 1);
            let none = routed.EnumAudioEndpoints(eRender, DEVICE_STATE(2)).unwrap();
            assert_eq!(none.GetCount().unwrap(), 0);
            // Capture is the host's.
            assert!(
                routed
                    .GetDefaultAudioEndpoint(EDataFlow(1), ERole(0))
                    .is_err()
            );
        }
    }

    #[test]
    fn the_hosts_enumerator_always_has_a_render_endpoint() {
        use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
        let mut out = std::ptr::null_mut();
        // SAFETY: COM calls with valid arguments.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            let hr = CoCreateInstance(
                &MMDeviceEnumerator,
                std::ptr::null_mut(),
                CLSCTX(0x17), // CLSCTX_ALL
                &IMMDeviceEnumerator::IID,
                &mut out,
            );
            if hr.is_err() {
                return; // no audio service on this host
            }
            let routed = IMMDeviceEnumerator::from_raw(out);
            // The host's own when it has an output, else ours: never an error.
            assert!(
                routed
                    .EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE)
                    .is_ok()
            );
        }
    }

    #[test]
    fn the_device_makes_a_client_that_takes_float_48k() {
        let device: IMMDevice = SilentDevice.into();
        // SAFETY: COM calls on our own objects.
        unsafe {
            let client: ISpatialAudioClient = device.Activate(CLSCTX(1), None).unwrap();
            client.IsAudioObjectFormatSupported(&FORMAT).unwrap();
            let bad = WAVEFORMATEX {
                nSamplesPerSec: 44_100,
                ..FORMAT
            };
            assert!(client.IsAudioObjectFormatSupported(&bad).is_err());
            assert_eq!(client.GetMaxFrameCount(&FORMAT).unwrap(), FRAMES);
            assert_eq!(
                client
                    .GetSupportedAudioObjectFormatEnumerator()
                    .unwrap()
                    .GetCount()
                    .unwrap(),
                1
            );
            assert!(device.Activate::<IMMDevice>(CLSCTX(1), None).is_err());
        }
    }

    fn stream(event: HANDLE, max_dynamic: u32) -> ISpatialAudioObjectRenderStream {
        SilentStream::new(Request {
            event: event.0 as usize,
            max_dynamic,
            static_mask: STATIC_OBJECTS,
        })
        .into()
    }

    /// The title's loop: wait for the event, then update the objects.
    fn cycle(stream: &ISpatialAudioObjectRenderStream, event: HANDLE) {
        // SAFETY: COM calls on our own objects; the buffer is ours.
        unsafe {
            assert_eq!(WaitForSingleObject(event, 1000).0, 0, "no period signal");
            let (mut available, mut frames) = (0, 0);
            stream
                .BeginUpdatingAudioObjects(&mut available, &mut frames)
                .unwrap();
            assert_eq!(frames, FRAMES);
            let object = stream
                .ActivateSpatialAudioObject(AudioObjectType_Dynamic)
                .unwrap();
            let (mut buffer, mut length) = (std::ptr::null_mut(), 0);
            object.GetBuffer(&mut buffer, &mut length).unwrap();
            assert_eq!(length, FRAMES * 4);
            buffer.write_bytes(0x7f, length as usize);
            stream.EndUpdatingAudioObjects().unwrap();
        }
    }

    #[test]
    fn the_stream_signals_every_period_and_restarts() {
        // SAFETY: COM calls on our own objects.
        unsafe {
            let event = CreateEventW(None, false, false, None).unwrap();
            let stream = stream(event, 4);
            for _ in 0..2 {
                stream.Start().unwrap();
                for _ in 0..20 {
                    cycle(&stream, event);
                }
                stream.Stop().unwrap();
                // A stopped stream signals nothing (one period may be in flight).
                let _ = WaitForSingleObject(event, 50);
                assert_ne!(
                    WaitForSingleObject(event, 50).0,
                    0,
                    "signalled while stopped"
                );
            }
            stream.Reset().unwrap();
        }
    }

    #[test]
    fn dynamic_objects_are_counted_and_released() {
        // SAFETY: COM calls on our own objects.
        unsafe {
            let event = CreateEventW(None, false, false, None).unwrap();
            let stream = stream(event, 2);
            let a = stream
                .ActivateSpatialAudioObject(AudioObjectType_Dynamic)
                .unwrap();
            let _b = stream
                .ActivateSpatialAudioObject(AudioObjectType_Dynamic)
                .unwrap();
            assert!(
                stream
                    .ActivateSpatialAudioObject(AudioObjectType_Dynamic)
                    .is_err()
            );
            drop(a);
            assert!(
                stream
                    .ActivateSpatialAudioObject(AudioObjectType_Dynamic)
                    .is_ok()
            );
            // A static object the title did not ask for.
            assert!(
                stream
                    .ActivateSpatialAudioObject(AudioObjectType(1 << 20))
                    .is_err()
            );
        }
    }
}
