# Collects CPU profiles of the engine with AMD uProf.
#
# Runs UNELEVATED - uProf's `collect` does not need administrator rights.
#
#   cd <repo root>
#   powershell -ExecutionPolicy Bypass -File engine\tools\profile.ps1
#
# Two hard-won notes about the uProf CLI:
#   - Never run `AMDuProfCLI <cmd> --help` from a script or a piped shell. Its
#     help is PAGED and blocks forever on "Press any key", reading the console
#     directly, so redirecting stdin does not help.
#   - `report` takes NO -o. Pass the session DIRECTORY to -i and it writes
#     report.csv inside it. Passing -o fails with a misleading error.
#
# Two workloads, because the engine and the agents are separate questions:
#   greedy    - a cheap agent, so most samples land in engine code
#   tactician - the search agent, ~97% of whose time is its own rollouts
#
# Both are sized to ~20s at uProf's default 1 kHz, so ~20k samples each - plenty
# to rank hot functions and enough to trust anything above ~1%. Game counts are
# calibrated to the CURRENT engine speed; the M8b/perf work has already made
# these 2.7x cheaper once, so re-check the reported Profile Duration and scale
# if a run comes in much under 20s.

$ErrorActionPreference = "Continue"

$uprof = "C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe"
$repo  = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$exe   = Join-Path $repo "engine\build\oo_run.exe"
$out   = Join-Path $repo "engine\prof"

if (-not (Test-Path $uprof)) { Write-Error "uProf CLI not found at $uprof"; exit 1 }
if (-not (Test-Path $exe))   { Write-Error "engine not built: $exe"; exit 1 }

New-Item -ItemType Directory -Force -Path $out | Out-Null

function Collect($name, $exeArgs) {
    $dir = Join-Path $out $name
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    Write-Host ""
    Write-Host "== collecting '$name' ==" -ForegroundColor Cyan
    Write-Host "   $exe $exeArgs"
    # tbp = time-based sampling: plain "where is the wall clock going".
    & $uprof collect --config tbp -o $dir $exe @exeArgs 2>&1 |
        Tee-Object -FilePath (Join-Path $out "$name.collect.log")

    # -i takes the SESSION DIRECTORY, not the .prd inside it, and -o is rejected
    # outright; the report lands as report.csv in that directory. --detail adds
    # the per-source-line breakdown, which is where the diagnosis actually lives.
    & $uprof report -i $dir --detail --cutoff 40 2>&1 |
        Tee-Object -FilePath (Join-Path $out "$name.report.log")
}

Collect "engine" @("--agent","greedy","--games","16000","--seed","1","--threads","1")
Collect "agent"  @("--agent","tactician","--games","1000","--seed","1","--threads","1")

Write-Host ""
Write-Host "Done. Results under $out" -ForegroundColor Green
Get-ChildItem -Recurse -File $out | Select-Object FullName,Length | Format-Table -AutoSize
