# Collects CPU profiles of the engine with AMD uProf.
#
# MUST RUN ELEVATED. uProf's sampling needs its kernel driver, so `collect`
# triggers a UAC prompt; from a non-elevated shell it hangs on a prompt that
# never appears. Right-click PowerShell -> Run as administrator, then:
#
#   cd C:\Users\Natha\Desktop\repos\oaths-and-omens
#   powershell -ExecutionPolicy Bypass -File engine\tools\profile.ps1
#
# Two workloads, because the engine and the agents are separate questions:
#   greedy    - a cheap agent, so most samples land in engine code
#   tactician - the search agent, ~97% of whose time is its own rollouts
#
# Both are sized to ~20s, which at uProf's default 1 kHz is ~20k samples each -
# plenty to rank hot functions, and enough to trust anything above ~1%.

$ErrorActionPreference = "Continue"

$uprof = "C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe"
$repo  = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$exe   = Join-Path $repo "engine\build\oo_run.exe"
$out   = Join-Path $repo "engine\prof"

if (-not (Test-Path $uprof)) { Write-Error "uProf CLI not found at $uprof"; exit 1 }
if (-not (Test-Path $exe))   { Write-Error "engine not built: $exe"; exit 1 }

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { Write-Error "Not elevated - run this from an Administrator PowerShell."; exit 1 }

New-Item -ItemType Directory -Force -Path $out | Out-Null

# Captured so the exact supported flags are on record - `collect --help` cannot be
# read from a non-elevated shell, and the flags below are best-effort until it is.
Write-Host "== capturing uProf CLI help =="
& $uprof collect --help  2>&1 | Out-File -Encoding utf8 (Join-Path $out "help_collect.txt")
& $uprof report  --help  2>&1 | Out-File -Encoding utf8 (Join-Path $out "help_report.txt")

function Collect($name, $exeArgs) {
    $dir = Join-Path $out $name
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    Write-Host ""
    Write-Host "== collecting '$name' ==" -ForegroundColor Cyan
    Write-Host "   $exe $exeArgs"
    # tbp = time-based sampling: plain "where is the wall clock going".
    & $uprof collect --config tbp -o $dir $exe @exeArgs 2>&1 |
        Tee-Object -FilePath (Join-Path $out "$name.collect.log")

    # The session lands in a timestamped subdirectory; find its .caperf/.prd.
    $session = Get-ChildItem -Path $dir -Recurse -Include *.caperf,*.prd -ErrorAction SilentlyContinue |
               Select-Object -First 1
    if (-not $session) { Write-Warning "no session file produced under $dir"; return }
    Write-Host "   session: $($session.FullName)"
    & $uprof report -i $session.FullName -o (Join-Path $out "$name.report") 2>&1 |
        Tee-Object -FilePath (Join-Path $out "$name.report.log")
}

Collect "engine" @("--agent","greedy","--games","6000","--seed","1")
Collect "agent"  @("--agent","tactician","--games","600","--seed","1")

Write-Host ""
Write-Host "Done. Results under $out" -ForegroundColor Green
Get-ChildItem -Recurse -File $out | Select-Object FullName,Length | Format-Table -AutoSize
