# Windows one-shot setup: downloads COCO val2017 + SAM checkpoint in parallel.
# Run from repo root in Anaconda Prompt:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# Optional: pass SAM model size
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -SamModel vit_l

param(
    [ValidateSet("vit_b","vit_l","vit_h")]
    [string]$SamModel = "vit_b"
)

$SamUrls = @{
    "vit_b" = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    "vit_l" = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    "vit_h" = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}
$SamUrl  = $SamUrls[$SamModel]
$SamFile = "checkpoints\" + (Split-Path $SamUrl -Leaf)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " HLCV Project Setup (Windows)"                               -ForegroundColor Cyan
Write-Host " SAM model: $SamModel"                                       -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ── 1. Directories ────────────────────────────────────────────────
Write-Host "`n[1/4] Creating directories..."
$dirs = @(
    "data\coco\val2017",
    "data\coco\annotations",
    "data\subsets",
    "checkpoints",
    "results",
    "figures",
    "notebooks",
    "report"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
Write-Host "      Done."

# ── 2. Python dependencies ────────────────────────────────────────
Write-Host "`n[2/4] Installing Python dependencies..."
pip install -q -r requirements.txt
pip install -q git+https://github.com/facebookresearch/segment-anything.git
pip install -q -e .
Write-Host "      Done."

# ── 3. Parallel downloads ─────────────────────────────────────────
Write-Host "`n[3/4] Downloading COCO val2017 + SAM checkpoint in parallel..."
Write-Host "      This may take several minutes depending on your connection."

function Download-File {
    param([string]$Url, [string]$Dest, [string]$Label)
    if (Test-Path $Dest) {
        Write-Host "      [skip] $Label already exists."
    } else {
        Write-Host "      Downloading $Label ..."
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($Url, (Resolve-Path ".").Path + "\" + $Dest)
        Write-Host "      [done] $Label"
    }
}

# Resolve absolute paths before launching jobs (Start-Job runs in a new process
# whose working directory defaults to $HOME\Documents, not the project root)
$ProjectRoot = $PSScriptRoot
$CocoImgDest  = Join-Path $ProjectRoot "data\coco\val2017.zip"
$CocoAnnDest  = Join-Path $ProjectRoot "data\coco\annotations_trainval2017.zip"
$SamFileDest  = Join-Path $ProjectRoot $SamFile

# Launch three downloads as background jobs
$job1 = Start-Job -ScriptBlock {
    param($u,$d,$l) Invoke-WebRequest -Uri $u -OutFile $d -UseBasicParsing
    Write-Output "done: $l"
} -ArgumentList "http://images.cocodataset.org/zips/val2017.zip", $CocoImgDest, "COCO images"

$job2 = Start-Job -ScriptBlock {
    param($u,$d,$l) Invoke-WebRequest -Uri $u -OutFile $d -UseBasicParsing
    Write-Output "done: $l"
} -ArgumentList "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", $CocoAnnDest, "COCO annotations"

$job3 = Start-Job -ScriptBlock {
    param($u,$d,$l) Invoke-WebRequest -Uri $u -OutFile $d -UseBasicParsing
    Write-Output "done: $l"
} -ArgumentList $SamUrl, $SamFileDest, "SAM $SamModel checkpoint"

Write-Host "      Waiting for downloads to complete (check Task Manager for progress)..."
$job1,$job2,$job3 | Wait-Job | Receive-Job
$job1,$job2,$job3 | Remove-Job

# ── 4. Extract ────────────────────────────────────────────────────
Write-Host "`n[4/4] Extracting archives..."

if (Test-Path $CocoImgDest) {
    Write-Host "      Extracting COCO images..."
    Expand-Archive -Path $CocoImgDest -DestinationPath (Join-Path $ProjectRoot "data\coco\") -Force
    Write-Host "      [done] Images → data\coco\val2017\"
}

if (Test-Path $CocoAnnDest) {
    Write-Host "      Extracting COCO annotations..."
    Expand-Archive -Path $CocoAnnDest -DestinationPath (Join-Path $ProjectRoot "data\coco\") -Force
    Write-Host "      [done] Annotations → data\coco\annotations\"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host " Next steps (run in Anaconda Prompt with hlcv env active):"
Write-Host ""
Write-Host "   1. Build subset:"
Write-Host "      python src\dataset\coco_loader.py --n-images 200 --output data\subsets\subset_200.json"
Write-Host ""
Write-Host "   2. Run evaluations:"
Write-Host "      python src\experiments\run_maskrcnn.py   --subset data\subsets\subset_200.json --device cuda"
Write-Host "      python src\experiments\run_mask2former.py --subset data\subsets\subset_200.json --device cuda"
Write-Host "      python src\experiments\run_sam.py         --subset data\subsets\subset_200.json --checkpoint $SamFile --device cuda"
Write-Host ""
Write-Host "   3. Generate plots:"
Write-Host "      python src\visualization\plot.py --sam results\sam_results.json --maskrcnn results\maskrcnn_results.json --mask2former results\mask2former_results.json --output figures\"
Write-Host "============================================================" -ForegroundColor Green
