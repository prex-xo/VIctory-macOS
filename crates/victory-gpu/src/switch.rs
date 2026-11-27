//! The `VICTORY_*` on/off switches (`VICTORY_D3D12_DEBUG=1` and its siblings).
#![allow(dead_code)]

use std::ffi::OsStr;

/// Whether a switch's value turns it on: it is set, and is not `0`.
///
/// A switch used to be on whenever its variable existed, so `VICTORY_D3D12_WARP=0` (typed to switch it off, or left
/// over from comparing two runs) still selected the software adapter. Any other value keeps the old meaning.
pub(crate) fn value_is_on(value: Option<&OsStr>) -> bool {
    value.is_some_and(|v| v != "0")
}

/// Whether an on-by-default switch's value leaves it on: anything except `0`.
fn value_is_on_by_default(value: Option<&OsStr>) -> bool {
    value != Some(OsStr::new("0"))
}

/// Whether the environment variable `name` switches something on.
pub(crate) fn on(name: &str) -> bool {
    value_is_on(std::env::var_os(name).as_deref())
}

/// Whether the environment variable `name` leaves an on-by-default switch on.
pub(crate) fn on_by_default(name: &str) -> bool {
    value_is_on_by_default(std::env::var_os(name).as_deref())
}

#[cfg(test)]
mod tests {
    use super::{on_by_default, value_is_on, value_is_on_by_default};
    use std::ffi::OsStr;

    #[test]
    fn unset_is_off() {
        assert!(!value_is_on(None));
    }

    #[test]
    fn zero_is_off() {
        assert!(!value_is_on(Some(OsStr::new("0"))));
    }

    #[test]
    fn one_is_on() {
        assert!(value_is_on(Some(OsStr::new("1"))));
    }

    #[test]
    fn other_values_keep_their_old_meaning() {
        // Documented as `=1`; anything else that was on before stays on: only `0` is new.
        for value in ["", "true", "yes", "2", "00", " 0", "0 "] {
            assert!(value_is_on(Some(OsStr::new(value))), "{value:?}");
        }
    }

    #[test]
    fn on_by_default_only_zero_is_off() {
        assert!(value_is_on_by_default(None));
        assert!(!value_is_on_by_default(Some(OsStr::new("0"))));
        assert!(value_is_on_by_default(Some(OsStr::new("1"))));
        assert!(value_is_on_by_default(Some(OsStr::new(""))));
        assert!(on_by_default("VICTORY_TEST_DEFINITELY_UNSET_SWITCH"));
    }
}
