use std::process::{Command, Output};

fn assert_missing_path(output: Output) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "scan succeeded: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(stderr.contains("missing shader input"), "{stderr}");
}

#[test]
fn shader_info_fails_for_a_missing_explicit_input() {
    let inputs = tempfile::tempdir().unwrap();
    let missing = inputs.path().join("missing shader input");

    let output = Command::new(env!("CARGO_BIN_EXE_victory"))
        .arg("shader-info")
        .args([inputs.path(), &missing])
        .output()
        .unwrap();
    assert_missing_path(output);
}

#[test]
fn shader_info_fails_for_a_missing_configured_input() {
    let inputs = tempfile::tempdir().unwrap();
    std::fs::write(
        inputs.path().join("victory.toml"),
        "[paths]\ngame_dir = 'missing shader input'\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_victory"))
        .arg("shader-info")
        .current_dir(inputs.path())
        .env_remove("VICTORY_CONFIG")
        .env_remove("VICTORY_GAME_DIR")
        .env_remove("VICTORY_TEST_TITLES")
        .env("VICTORY_DATA_DIR", inputs.path().join("data"))
        .env("VICTORY_CACHE_DIR", inputs.path().join("cache"))
        .output()
        .unwrap();
    assert_missing_path(output);
}

#[test]
fn shader_info_accepts_an_existing_file_with_no_shaders() {
    let inputs = tempfile::tempdir().unwrap();
    let valid = inputs.path().join("notes.txt");
    std::fs::write(&valid, "no shaders").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_victory"))
        .arg("shader-info")
        .arg(&valid)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("0 (in 0 files)"), "{stdout}");
}

#[cfg(windows)]
#[test]
fn translate_fails_for_a_missing_input_before_setting_up_the_host_or_cache() {
    let inputs = tempfile::tempdir().unwrap();
    let missing = inputs.path().join("missing shader input");
    let cache = inputs.path().join("cache");
    for mode in ["--pipelines", "--cache"] {
        let output = Command::new(env!("CARGO_BIN_EXE_victory"))
            .args(["translate", mode])
            .arg(&missing)
            .env("VICTORY_DATA_DIR", inputs.path().join("data"))
            .env("VICTORY_CACHE_DIR", &cache)
            .output()
            .unwrap();
        assert_missing_path(output);
        assert!(!cache.exists());
    }
}
