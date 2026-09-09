# Reblesses the seed-derived golden files from the current engine (PLAN.md §3.4).
#
#   powershell -ExecutionPolicy Bypass -File engine\tools\rewrite_goldens.ps1 -Check
#   powershell -ExecutionPolicy Bypass -File engine\tools\rewrite_goldens.ps1 -Apply
#
# -Check   rewrites to temporary files and reports whether git would see a change.
#          With the engine unmodified this MUST report every file unchanged - that
#          is what proves the rewrite path is faithful rather than merely
#          self-consistent. Run it before changing engine behaviour, not after.
#
# -Apply   overwrites the golden files in place.
#
# turn_traces uses --record, not --rewrite, and that difference matters. Rewriting
# keeps the recorded decision traces, which only works while the engine still ASKS
# for the same decisions; a change to the dice or the movement rules alters battle
# round counts and traces get DROPPED. The RNG swap alone eroded the corpus from
# 176 cases to 80. --record regenerates the whole thing from seeds instead, so the
# deepest test in the suite survives deliberate change rather than decaying with
# every one.
#
# The other rewriting tests keep their recorded inputs and recompute only the
# expected outputs. test_setup is different: its terrain half rewrites, but its
# setup half RE-RECORDS from the seed with native agents, because a changed RNG
# produces a different board and the recorded placements point at hexes that no
# longer exist. See tests/test_setup.cpp.
#
# legal_cases is here too, and that is not about the RNG: it records the
# movement/cavalry MASKS, so any rules change to legality invalidates it. §11.1
# did exactly that.
#
# Files NOT touched, because nothing reaches them: grid_golden (pure geometry),
# phase_cases (no Rng is even constructed), and buy_scenarios (hand-built inputs
# AND hand-reasoned expectations - a specification, not a recording; reblessing
# it would just make it agree with whatever the engine now does).
#
# rng_golden and rng_stress are deliberately absent: they exist only to pin
# CPython bit-compatibility, so the RNG swap retires them rather than reblessing
# them.

[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$Apply
)

if (-not $Check -and -not $Apply) {
    Write-Host "specify -Check (dry run, the default thing you want) or -Apply"
    exit 2
}

$ErrorActionPreference = "Stop"
$repo  = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build = Join-Path $repo "engine\build"
$data  = Join-Path $repo "engine\tests\data"

$targets = @(
    @{ exe = "test_agents";   file = "agent_games.txt" },
    @{ exe = "test_replay";   file = "replay_hashes.txt" },
    @{ exe = "test_movement"; file = "movement_scenarios.txt" },
    @{ exe = "test_turn";     file = "turn_traces.txt"; flag = "--record" },
    @{ exe = "test_setup";    file = "setup_cases.txt" },
    @{ exe = "test_buy";      file = "legal_cases.txt"; extra = "tests\data\buy_scenarios.txt" }
)

$changed = 0
foreach ($t in $targets) {
    $exe = Join-Path $build ($t.exe + ".exe")
    if (-not (Test-Path $exe)) { Write-Error "not built: $exe"; exit 1 }
    $golden = Join-Path $data $t.file
    $tmp    = Join-Path ([System.IO.Path]::GetTempPath()) ("oo_" + $t.file)

    # test_buy takes two positional files; the second is a specification, not a
    # recording, so it is passed through but never reblessed.
    $flag = if ($t.ContainsKey("flag")) { $t.flag } else { "--rewrite" }
    if ($t.ContainsKey("extra")) {
        & $exe $golden (Join-Path $repo ("engine\\" + $t.extra)) $flag $tmp | Out-Null
    } else {
        & $exe $golden $flag $tmp | Out-Null
    }
    if ($LASTEXITCODE -ne 0) { Write-Error "$($t.exe) --rewrite failed"; exit 1 }

    # Compare the CONTENT of the two files directly, normalising line endings.
    #
    # An earlier version compared with `git diff` and restored with
    # `git checkout`. Both were wrong the moment the working tree was dirty: git
    # diffs against HEAD, not against the file as it was before this run, so a
    # -Check during an uncommitted rebless reported everything as changed and
    # then REVERTED the rebless. This depends on nothing but the two files.
    $before = [System.IO.File]::ReadAllText($golden) -replace "`r`n", "`n"
    $after  = [System.IO.File]::ReadAllText($tmp)    -replace "`r`n", "`n"
    $differs = ($before -ne $after)

    if ($differs) {
        $oldLines = ($before -split "`n").Count
        $newLines = ($after  -split "`n").Count
        Write-Host ("  {0,-26} CHANGED  ({1} -> {2} lines)" -f $t.file, $oldLines, $newLines) -ForegroundColor Yellow
        $changed++
    } else {
        Write-Host ("  {0,-26} unchanged" -f $t.file) -ForegroundColor DarkGray
    }

    if ($Apply -and $differs) {
        Copy-Item $tmp $golden -Force
    }
}

Write-Host ""
if ($Check) {
    if ($changed -eq 0) {
        Write-Host "All goldens reproduce exactly. The rewrite path is faithful." -ForegroundColor Green
    } else {
        Write-Host "$changed file(s) would change." -ForegroundColor Yellow
        Write-Host "If the engine is UNMODIFIED this is a bug in the rewrite path, not a rebless." -ForegroundColor Yellow
    }
} else {
    Write-Host "$changed file(s) reblessed. Review the diff before committing." -ForegroundColor Green
}
