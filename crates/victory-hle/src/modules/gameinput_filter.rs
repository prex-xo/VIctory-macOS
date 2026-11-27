//! Opt-in input filtering for low-frame-rate play and raw controller sticks
//! (`VICTORY_INPUT_LATCH=1`, `VICTORY_STICK_DEADZONE=<0..1>`), all off by
//! default.
//!
//! - `VICTORY_INPUT_LATCH=1`: when the title polls `GetCurrentReading` only a
//!   few times per second, short button taps that start and end between two
//!   polls would otherwise vanish from the latest state. Latching holds each
//!   unobserved press (and release, plus a subsequent re-press queued for the
//!   following reading) until a reading is handed out, keeps each trigger's
//!   peak since the last hand-out, and passes thumbsticks through live.
//!   Readings handed out within [`LATCH_HOLD`] of each other see consistent
//!   state; if no reading is requested for [`LATCH_MAX`], the latch releases so
//!   buttons never remain stuck.
//! - `VICTORY_STICK_DEADZONE=<0..1>`: applies a radial deadzone with smooth
//!   linear rescaling to host gamepad thumbsticks before readings are recorded.

use super::{FALLBACK, PADS, PadState};

/// Devices with a latch: the host gamepads plus the keyboard stand-in pad.
const LATCHED: usize = FALLBACK + 1;
const _: () = assert!(FALLBACK == PADS);

/// Readings handed out within this window (µs) keep the same latched buttons.
const LATCH_HOLD: u64 = 15_000;
/// Maximum time (µs) to hold a latch without a reading hand-out.
const LATCH_MAX: u64 = 2_000_000;

#[derive(Clone, Copy, Debug, Default)]
struct Latch {
    /// Buttons currently exposed in readings.
    shown: u32,
    /// `shown` at the last reading hand-out (what the title last observed).
    seen: u32,
    /// Since the last hand-out: buttons observed down, observed up, and
    /// observed down again after being up.
    down: u32,
    up: u32,
    again: u32,
    /// Buttons that were down in `seen`, released, and pressed again before the
    /// title polled: shown up in the next reading, then down in the one after.
    queued: u32,
    /// Trigger peaks observed since the last hand-out, and what readings show.
    peak: [f32; 2],
    triggers: [f32; 2],
    /// While holding the handed-out state stable: timestamp (µs) of hand-out.
    holding: Option<u64>,
    /// Timestamp (µs) of the last reading hand-out.
    handed: u64,
}

impl Latch {
    fn hand_out(&mut self, now: u64) {
        if self.holding.is_some() {
            return;
        }
        self.queued &= !self.shown;
        self.seen = self.shown;
        (self.down, self.up, self.again) = (0, 0, 0);
        self.peak = [0.0; 2];
        self.holding = Some(now);
        self.handed = now;
    }

    fn apply(&mut self, pad: PadState, now: u64, hold: u64) -> PadState {
        if self.holding.is_none() && now.saturating_sub(self.handed) >= LATCH_MAX {
            self.hand_out(now);
        }
        let host = pad.buttons;
        self.again |= self.up & host;
        self.down |= host;
        self.up |= !host;
        self.peak[0] = self.peak[0].max(pad.left_trigger);
        self.peak[1] = self.peak[1].max(pad.right_trigger);
        if self
            .holding
            .is_some_and(|at| now.saturating_sub(at) >= hold)
        {
            self.holding = None;
        }
        if self.holding.is_none() {
            let seen = self.seen;
            self.queued |= seen & self.up & self.again;
            self.shown = (seen & !self.up) | (!seen & (self.down | self.queued));
            self.triggers = self.peak;
        }
        PadState {
            buttons: self.shown,
            left_trigger: self.triggers[0],
            right_trigger: self.triggers[1],
            ..pad
        }
    }
}

fn parse_latch(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .is_some_and(|v| !matches!(v, "" | "0" | "off" | "false"))
}

fn parse_deadzone(value: Option<&str>) -> Result<f32, String> {
    match value.map(str::trim).filter(|v| !v.is_empty()) {
        None | Some("0" | "off" | "false") => Ok(0.0),
        Some(v) => match v.parse::<f32>() {
            Ok(0.0) => Ok(0.0),
            Ok(zone) if zone.is_finite() && zone > 0.0 && zone < 1.0 => Ok(zone),
            _ => Err(v.to_owned()),
        },
    }
}

/// Applies a radial deadzone of `zone` in `[0, 1)` to a stick vector `(x, y)`,
/// smoothly rescaling magnitudes in `(zone, 1]` back onto `(0, 1]`.
pub(super) fn radial_deadzone(x: f32, y: f32, zone: f32) -> (f32, f32) {
    if zone <= 0.0 {
        return (x, y);
    }
    let len = x.hypot(y);
    if len <= zone || len == 0.0 {
        return (0.0, 0.0);
    }
    let rescaled = ((len - zone) / (1.0 - zone)).min(1.0);
    let scale = rescaled / len;
    (x * scale, y * scale)
}

/// Opt-in input filters; `Default` disables all filtering.
#[derive(Debug, Default)]
pub(super) struct InputFilter {
    /// `Some(hold_us)` when button/trigger latching is enabled.
    latch: Option<u64>,
    latches: [Latch; LATCHED],
    /// Radial thumbstick deadzone in `[0, 1)`; `0.0` disables filtering.
    deadzone: f32,
}

impl InputFilter {
    pub(super) fn from_env() -> Self {
        let mut filter = Self::default();
        if parse_latch(std::env::var("VICTORY_INPUT_LATCH").ok().as_deref()) {
            tracing::info!(
                hold_ms = LATCH_HOLD / 1000,
                "button latch enabled (VICTORY_INPUT_LATCH)"
            );
            filter.latch = Some(LATCH_HOLD);
        }
        match parse_deadzone(std::env::var("VICTORY_STICK_DEADZONE").ok().as_deref()) {
            Ok(zone) if zone > 0.0 => {
                tracing::info!(zone, "stick radial deadzone enabled (VICTORY_STICK_DEADZONE)");
                filter.deadzone = zone;
            }
            Ok(_) => {}
            Err(value) => tracing::warn!(
                value,
                "VICTORY_STICK_DEADZONE expects a float in [0, 1); ignoring"
            ),
        }
        filter
    }

    /// Applies the configured radial deadzone to all connected host pads.
    pub(super) fn apply_deadzone(&self, pads: &mut [Option<PadState>; PADS]) {
        if self.deadzone <= 0.0 {
            return;
        }
        for pad in pads.iter_mut().flatten() {
            (pad.left_x, pad.left_y) = radial_deadzone(pad.left_x, pad.left_y, self.deadzone);
            (pad.right_x, pad.right_y) = radial_deadzone(pad.right_x, pad.right_y, self.deadzone);
        }
    }

    /// Transforms device `index`'s gamepad state through the latch if enabled.
    pub(super) fn latch(
        &mut self,
        index: usize,
        pad: Option<PadState>,
        now: u64,
    ) -> Option<PadState> {
        let (Some(hold), Some(latch)) = (self.latch, self.latches.get_mut(index)) else {
            return pad;
        };
        let Some(pad) = pad else {
            *latch = Latch {
                handed: now,
                ..Latch::default()
            };
            return None;
        };
        Some(latch.apply(pad, now, hold))
    }

    /// Records that a reading for device `index` was handed to the title at `now`.
    pub(super) fn handed(&mut self, index: usize, now: u64) {
        if self.latch.is_some()
            && let Some(latch) = self.latches.get_mut(index)
        {
            latch.hand_out(now);
        }
    }
}

#[cfg(test)]
pub(super) fn for_tests(latch_hold: Option<u64>, deadzone: f32) -> InputFilter {
    InputFilter {
        latch: latch_hold,
        latches: Default::default(),
        deadzone,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u32 = 4;

    fn pad(buttons: u32, left_trigger: f32) -> Option<PadState> {
        Some(PadState {
            buttons,
            left_trigger,
            left_x: 0.25,
            ..Default::default()
        })
    }

    fn buttons(filter: &mut InputFilter, host: u32, now: u64) -> u32 {
        filter.latch(0, pad(host, 0.0), now).unwrap().buttons
    }

    #[test]
    fn env_parsing_defaults_off_and_validates_bounds() {
        for off in [None, Some(""), Some("0"), Some(" 0 "), Some("off"), Some("false")] {
            assert!(!parse_latch(off), "{off:?}");
        }
        for on in [Some("1"), Some(" 1 "), Some("on"), Some("true")] {
            assert!(parse_latch(on), "{on:?}");
        }
        for off in [None, Some(""), Some("0"), Some("0.0"), Some("off"), Some("false")] {
            assert_eq!(parse_deadzone(off), Ok(0.0), "{off:?}");
        }
        assert_eq!(parse_deadzone(Some("0.15")), Ok(0.15));
        assert_eq!(parse_deadzone(Some(" 0.5 ")), Ok(0.5));
        for bad in ["-0.1", "1.0", "1.5", "NaN", "inf", "abc"] {
            assert!(parse_deadzone(Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn double_tap_between_polls_delivers_release_then_second_press() {
        let mut filter = for_tests(Some(0), 0.0);
        assert_eq!(buttons(&mut filter, A, 10), A);
        filter.handed(0, 11);
        for (host, t) in [(0, 20), (A, 30), (0, 40)] {
            assert_eq!(buttons(&mut filter, host, t), 0);
        }
        filter.handed(0, 41);
        assert_eq!(buttons(&mut filter, 0, 50), A, "queued second press");
        filter.handed(0, 51);
        assert_eq!(buttons(&mut filter, 0, 60), 0);
    }

    #[test]
    fn reads_within_hold_window_agree_then_release() {
        let mut filter = for_tests(Some(LATCH_HOLD), 0.0);
        assert_eq!(buttons(&mut filter, A, 10), A);
        filter.handed(0, 1_000);
        assert_eq!(buttons(&mut filter, 0, 9_000), A, "stable inside hold window");
        filter.handed(0, 9_500);
        assert_eq!(buttons(&mut filter, 0, 17_000), 0, "updated after hold window");
    }

    #[test]
    fn unread_latch_expires_after_max_interval() {
        let mut filter = for_tests(Some(0), 0.0);
        assert_eq!(buttons(&mut filter, A, 10), A);
        assert_eq!(buttons(&mut filter, 0, 20), A);
        assert_eq!(buttons(&mut filter, 0, LATCH_MAX + 1), 0);
    }

    #[test]
    fn triggers_keep_peak_sticks_stay_live_and_disconnect_resets() {
        let mut filter = for_tests(Some(0), 0.0);
        filter.latch(1, pad(0, 0.75), 10);
        let shown = filter.latch(1, pad(0, 0.25), 20).unwrap();
        assert_eq!((shown.left_trigger, shown.left_x), (0.75, 0.25));
        filter.handed(1, 25);
        assert_eq!(filter.latch(1, pad(0, 0.25), 30).unwrap().left_trigger, 0.25);

        filter.latch(1, pad(A, 0.0), 40);
        assert_eq!(filter.latch(1, None, 50), None);
        assert_eq!(filter.latch(1, pad(0, 0.0), 60).unwrap().buttons, 0);
    }

    #[test]
    fn radial_deadzone_zeroes_inside_rescales_smoothly_and_is_symmetric() {
        let zone = 0.25;
        // Inside and on boundary -> 0.
        assert_eq!(radial_deadzone(0.0, 0.0, zone), (0.0, 0.0));
        assert_eq!(radial_deadzone(0.25, 0.0, zone), (0.0, 0.0));
        assert_eq!(radial_deadzone(0.15, -0.15, zone), (0.0, 0.0));

        // Midpoint between zone and 1.0 rescales to 0.5.
        let (x, y) = radial_deadzone(0.0, 0.625, zone);
        assert!(x == 0.0 && (y - 0.5).abs() < 1e-6);

        // Full deflection stays 1.0; diagonal clamps to unit circle keeping direction.
        let (x, y) = radial_deadzone(0.0, -1.0, zone);
        assert!(x == 0.0 && (y + 1.0).abs() < 1e-6);
        let (x, y) = radial_deadzone(0.8, 0.8, zone);
        assert!((x - y).abs() < 1e-6 && (x.hypot(y) - 1.0).abs() < 1e-6);

        // Radial symmetry across angles at radius r = 0.625 (expected output r' = 0.5).
        let r = 0.625f32;
        for deg in [0.0f32, 30.0, 45.0, 90.0, 135.0, 210.0, 315.0] {
            let rad = deg.to_radians();
            let (ox, oy) = radial_deadzone(r * rad.cos(), r * rad.sin(), zone);
            assert!(
                (ox.hypot(oy) - 0.5).abs() < 1e-5,
                "angle {deg}: got magnitude {}",
                ox.hypot(oy)
            );
            assert!(
                (ox - 0.5 * rad.cos()).abs() < 1e-5 && (oy - 0.5 * rad.sin()).abs() < 1e-5,
                "angle {deg}: direction drifted"
            );
        }

        // Disabled by default: raw stick passes through unchanged.
        let mut pads = [pad(0, 0.0), None, None, None];
        InputFilter::default().apply_deadzone(&mut pads);
        assert_eq!(pads[0].unwrap().left_x, 0.25);
    }
}
