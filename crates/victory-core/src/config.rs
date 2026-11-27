//! Where the user's files live.
//!
//! Resolution order for each value: environment variable, then the config
//! file (`VICTORY_CONFIG`, else `./victory.toml`), then the default.
//! See `victory.example.toml` at the repository root.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::policy::{ensure_outside_repo, resolve as resolve_path};

#[derive(Debug, Default, Clone)]
pub struct Config {
    /// The user's dump of Grand Theft Auto VI.
    pub game_dir: Option<PathBuf>,
    /// Root for all derived data. Always outside the repository.
    pub cache_dir: PathBuf,
    /// What titles write and must keep (persistent local storage). Never
    /// inside the cache, which may be deleted at any time.
    pub data_dir: PathBuf,
    /// Other titles used to exercise platform-level code.
    pub test_titles: Vec<PathBuf>,
    /// The user's copy of AMD FidelityFX for PC (the folder with its
    /// upscaler DLL), which replaces a title's console build.
    pub fidelityfx_dir: Option<PathBuf>,
    /// The config file that was read, if any.
    pub source: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default)]
    paths: FilePaths,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FilePaths {
    game_dir: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    #[serde(default)]
    test_titles: Vec<PathBuf>,
    fidelityfx_dir: Option<PathBuf>,
}

impl Config {
    /// Loads configuration from the environment and the config file.
    pub fn load() -> Result<Self> {
        let explicit = explicit_config_path(|k| std::env::var_os(k));
        let path = explicit
            .clone()
            .unwrap_or_else(|| PathBuf::from("victory.toml"));
        let file = if path.is_file() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            Some((path, text))
        } else if let Some(p) = explicit {
            anyhow::bail!(
                "VICTORY_CONFIG points at {}, which does not exist",
                p.display()
            );
        } else {
            None
        };
        let (source, parsed) = match file {
            Some((path, text)) => {
                let parsed: FileConfig =
                    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
                (Some(path), parsed)
            }
            None => (None, FileConfig::default()),
        };
        Self::resolve(parsed.paths, source, |k| std::env::var_os(k))
    }

    fn resolve(
        file: FilePaths,
        source: Option<PathBuf>,
        env: impl Fn(&str) -> Option<std::ffi::OsString>,
    ) -> Result<Self> {
        let env_path = |k: &str| env(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        let test_titles = match env("VICTORY_TEST_TITLES") {
            Some(v) if !v.is_empty() => std::env::split_paths(&v)
                .filter(|p| !p.as_os_str().is_empty())
                .collect(),
            _ => file.test_titles,
        };
        let cache_dir = env_path("VICTORY_CACHE_DIR")
            .or(file.cache_dir)
            .map_or_else(|| default_dir("cache"), Ok)?;
        let data_dir = env_path("VICTORY_DATA_DIR")
            .or(file.data_dir)
            .map_or_else(|| default_dir("data"), Ok)?;
        ensure_data_outside_cache(&data_dir, &cache_dir)?;
        Ok(Config {
            game_dir: env_path("VICTORY_GAME_DIR").or(file.game_dir),
            cache_dir,
            data_dir,
            test_titles,
            fidelityfx_dir: env_path("VICTORY_FIDELITYFX_DIR").or(file.fidelityfx_dir),
            source,
        })
    }

    /// Returns `<cache>/<sub>`, creating it, after checking the asset policy.
    pub fn cache_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.cache_dir, sub)
    }

    /// Returns `<data>/<sub>`, creating it, after checking the asset policy.
    pub fn data_subdir(&self, sub: &str) -> Result<PathBuf> {
        let dir = self.data_dir.join(sub);
        ensure_data_outside_cache(&dir, &self.cache_dir)?;
        subdir(&self.data_dir, sub)
    }
}

/// Persistent title data must survive cache cleanup, so `data_dir` may not
/// be the cache directory or inside it. Both paths go through the
/// longest-existing-prefix resolver first, so symlinks and not-yet-created
/// leaf directories are handled; `starts_with` then compares components, so
/// a sibling like `cache-backup` is accepted.
fn ensure_data_outside_cache(data_dir: &Path, cache_dir: &Path) -> Result<()> {
    let resolved_data =
        resolve_path(data_dir).with_context(|| format!("resolving {}", data_dir.display()))?;
    let resolved_cache =
        resolve_path(cache_dir).with_context(|| format!("resolving {}", cache_dir.display()))?;
    // Wine can keep the caller's spelling when canonicalizing an existing path.
    // Ask the filesystem whether the ancestor at the cache depth is the cache.
    let ancestor: PathBuf = resolved_data
        .components()
        .take(resolved_cache.components().count())
        .collect();
    if resolved_data.starts_with(&resolved_cache)
        || (resolved_cache.try_exists()?
            && ancestor.try_exists()?
            && same_file::is_same_file(&ancestor, &resolved_cache)?)
    {
        anyhow::bail!(
            "data_dir {} (resolved {}) is equal to or inside cache_dir {} (resolved {}); \
             cache cleanup may delete it, so persistent title saves would be lost — \
             keep data_dir outside cache_dir",
            data_dir.display(),
            resolved_data.display(),
            cache_dir.display(),
            resolved_cache.display()
        );
    }
    Ok(())
}

/// The config file explicitly named by the environment, if any. An empty
/// `VICTORY_CONFIG` means unset, like the other path variables.
fn explicit_config_path(env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Option<PathBuf> {
    env("VICTORY_CONFIG")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn subdir(root: &Path, sub: &str) -> Result<PathBuf> {
    let dir = root.join(sub);
    ensure_outside_repo(&dir)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// `%LOCALAPPDATA%/victory/<name>` (elsewhere the XDG or home cache).
fn default_dir(name: &str) -> Result<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .context("cannot determine a user directory; set VICTORY_CACHE_DIR and VICTORY_DATA_DIR")?;
    Ok(base.join("victory").join(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn environment_overrides_file() {
        let file = FilePaths {
            game_dir: Some("from-file".into()),
            cache_dir: Some("cache-from-file".into()),
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_GAME_DIR").then(|| OsString::from("from-env"));
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-env"));
        assert_eq!(cfg.cache_dir, PathBuf::from("cache-from-file"));
    }

    #[test]
    fn empty_env_values_are_ignored() {
        let file = FilePaths {
            game_dir: Some("from-file".into()),
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_GAME_DIR").then(OsString::new);
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-file"));
    }

    #[test]
    fn empty_test_titles_env_is_ignored() {
        let file = FilePaths {
            test_titles: vec!["from-file".into()],
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_TEST_TITLES").then(OsString::new);
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.test_titles, vec![PathBuf::from("from-file")]);
    }

    #[test]
    fn test_titles_env_splits_paths() {
        let value = std::env::join_paths(["first", "second"]).unwrap();
        let file = FilePaths::default();
        let env = |k: &str| (k == "VICTORY_TEST_TITLES").then(|| value.clone());
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(
            cfg.test_titles,
            vec![PathBuf::from("first"), PathBuf::from("second")]
        );
    }

    #[test]
    fn empty_config_env_is_ignored() {
        let empty = |k: &str| (k == "VICTORY_CONFIG").then(OsString::new);
        assert_eq!(explicit_config_path(empty), None);
        let set = |k: &str| (k == "VICTORY_CONFIG").then(|| OsString::from("custom.toml"));
        assert_eq!(
            explicit_config_path(set),
            Some(PathBuf::from("custom.toml"))
        );
    }

    #[test]
    fn example_config_parses() {
        let text = include_str!("../../../victory.example.toml");
        let parsed: FileConfig = toml::from_str(text).unwrap();
        assert!(parsed.paths.game_dir.is_some());
    }

    #[test]
    fn cache_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        let cfg = Config {
            cache_dir: inside,
            ..Default::default()
        };
        assert!(cfg.cache_subdir("x").is_err());
    }

    #[test]
    fn data_inside_repo_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let cfg = Config {
            cache_dir: tmp.path().join("cache"),
            data_dir: inside,
            ..Default::default()
        };
        let err = cfg.data_subdir("x").unwrap_err();
        assert!(err.downcast_ref::<crate::PolicyError>().is_some(), "{err}");
    }

    #[test]
    fn persistent_data_stays_outside_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        let data = tmp.path().join("data");
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::create_dir_all(&data).unwrap();

        let resolve_with = |cache_dir: PathBuf, data_dir: PathBuf| {
            Config::resolve(
                FilePaths {
                    cache_dir: Some(cache_dir),
                    data_dir: Some(data_dir),
                    ..Default::default()
                },
                None,
                |_: &str| None,
            )
        };

        // Equal roots are rejected.
        assert!(resolve_with(cache.clone(), cache.clone()).is_err());

        // A data_dir nested inside cache_dir is rejected, and the error names both paths.
        let nested = cache.join("saves");
        let err = resolve_with(cache.clone(), nested.clone()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(&cache.display().to_string()), "{msg}");
        assert!(msg.contains(&nested.display().to_string()), "{msg}");

        // A symlink alias pointing into the cache is rejected, even behind `missing/..`.
        #[cfg(unix)]
        {
            let alias = tmp.path().join("alias");
            std::os::unix::fs::symlink(&cache, &alias).unwrap();
            assert!(resolve_with(cache.clone(), alias).is_err());
            let via_parent = tmp
                .path()
                .join("missing")
                .join("..")
                .join("alias")
                .join("saves");
            assert!(resolve_with(cache.clone(), via_parent.clone()).is_err());
            let cache_via_parent = tmp.path().join("missing").join("..").join("alias");
            assert!(resolve_with(cache_via_parent, cache.join("saves")).is_err());
        }

        // On case-insensitive volumes, differently cased roots that did not exist at
        // resolve time are still caught before `data_subdir` creates persistent storage.
        let probe_upper = tmp.path().join("CaseProbe");
        std::fs::create_dir(&probe_upper).unwrap();
        if tmp.path().join("caseprobe").exists() {
            let new_cache = tmp.path().join("NewCache");
            let new_data = tmp.path().join("newcache");
            let cfg = resolve_with(new_cache.clone(), new_data).unwrap();
            cfg.cache_subdir("shaders").unwrap();
            assert!(cfg.data_subdir("saves").is_err());
            assert!(!new_cache.join("saves").exists());
        } else {
            let lower = tmp.path().join("caseprobe");
            std::fs::create_dir(&lower).unwrap();
            let cfg = resolve_with(probe_upper, lower.clone()).unwrap();
            assert_eq!(cfg.data_subdir("saves").unwrap(), lower.join("saves"));
        }

        // If cache_dir lives inside data_dir, a data_subdir landing inside cache_dir is rejected.
        let nested_cache = data.join("00000000");
        let cfg = resolve_with(nested_cache.clone(), data.clone()).unwrap();
        assert!(cfg.data_subdir("00000000/storage").is_err());
        assert!(!nested_cache.join("storage").exists());

        // Sibling directories are accepted, even one named like the cache.
        assert!(resolve_with(cache.clone(), data.clone()).is_ok());
        let sibling = tmp.path().join("cache-backup");
        assert!(resolve_with(cache.clone(), sibling).is_ok());
    }
}
