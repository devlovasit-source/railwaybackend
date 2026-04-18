param(
    [string]$ImagePath,
    [string]$UserId = "user_demo_1",
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [switch]$SkipBoardSave
)

$ErrorActionPreference = "Stop"

function Write-Pass($msg) { Write-Host "[PASS] $msg" -ForegroundColor Green }
function Write-Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red }
function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }

function Try-Call {
    param(
        [scriptblock]$Action,
        [string]$Name
    )
    try {
        $result = & $Action
        Write-Pass $Name
        return @{ ok = $true; result = $result }
    } catch {
        Write-Fail "$Name :: $($_.Exception.Message)"
        return @{ ok = $false; result = $null }
    }
}

function Resolve-ImagePath {
    param([string]$InputPath)

    if ($InputPath -and (Test-Path -LiteralPath $InputPath)) {
        return (Resolve-Path -LiteralPath $InputPath).Path
    }

    if ($InputPath -and -not (Test-Path -LiteralPath $InputPath)) {
        Write-Info "Provided ImagePath not found: $InputPath"
    }

    $roots = @(
        [Environment]::GetFolderPath("MyPictures"),
        [Environment]::GetFolderPath("Desktop"),
        [Environment]::GetFolderPath("MyDocuments"),
        (Join-Path $env:USERPROFILE "Downloads")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

    $patterns = @("*.jpg", "*.jpeg", "*.png", "*.webp")
    $found = @()
    foreach ($r in $roots) {
        foreach ($pat in $patterns) {
            $found += Get-ChildItem -LiteralPath $r -Filter $pat -File -ErrorAction SilentlyContinue
        }
    }

    $pick = $found | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($pick) {
        Write-Info "Auto-selected latest image: $($pick.FullName)"
        return $pick.FullName
    }

    throw "No image found. Pass -ImagePath with a valid jpg/jpeg/png/webp file."
}

function Test-UrlReachable {
    param([string]$Url)
    if (-not $Url) { return $false }

    try {
        $request = [System.Net.HttpWebRequest]::Create($Url)
        $request.Method = "GET"
        $request.Timeout = 20000
        $request.ReadWriteTimeout = 20000
        $request.AllowAutoRedirect = $true
        $request.UserAgent = "AHVI-QA/1.0"
        $response = $request.GetResponse()
        $statusCode = [int]([System.Net.HttpWebResponse]$response).StatusCode
        $response.Close()
        return ($statusCode -ge 200 -and $statusCode -lt 400)
    } catch {
        return $false
    }
}

$ImagePath = Resolve-ImagePath -InputPath $ImagePath

Write-Info "AHVI Instant QA starting"
Write-Info "BaseUrl: $BaseUrl"
Write-Info "UserId: $UserId"
Write-Info "Image: $ImagePath"

# 1) Health
$healthCall = Try-Call -Name "Health endpoint" -Action {
    Invoke-RestMethod -Method Get -Uri "$BaseUrl/health"
}
if ($healthCall.ok) {
    Write-Info ("health response: " + ($healthCall.result | ConvertTo-Json -Compress))
}

# 2) OpenAPI route discovery
$openapiCall = Try-Call -Name "OpenAPI fetch" -Action {
    Invoke-RestMethod -Method Get -Uri "$BaseUrl/openapi.json"
}
if (-not $openapiCall.ok) {
    Write-Fail "Cannot continue without OpenAPI"
    exit 1
}

$paths = @($openapiCall.result.paths.PSObject.Properties.Name)

$analyzePath = $paths | Where-Object { $_ -match "wardrobe/capture/analyze" } | Select-Object -First 1
$savePath = $paths | Where-Object { $_ -match "wardrobe/capture/save-selected" } | Select-Object -First 1
$boardSaveCandidates = $paths | Where-Object { $_ -match "boards/.*/save|boards/save" -and $_ -notmatch "life/save" }
$boardSavePath = $boardSaveCandidates | Select-Object -First 1

if ($analyzePath) { Write-Pass "Found analyze path: $analyzePath" } else { Write-Fail "Analyze path not found" }
if ($savePath) { Write-Pass "Found save-selected path: $savePath" } else { Write-Fail "Save-selected path not found" }
if ($boardSavePath) {
    Write-Pass "Found board-save path: $boardSavePath"
} else {
    Write-Info "Board-save path not found (this check is optional)"
}

if (-not $analyzePath -or -not $savePath) {
    Write-Fail "Wardrobe capture routes missing. Stop."
    exit 1
}

# 3) Analyze
$imageBytes = [System.IO.File]::ReadAllBytes($ImagePath)
$imageBase64 = [System.Convert]::ToBase64String($imageBytes)

$analyzeBody = @{
    user_id = $UserId
    image_base64 = $imageBase64
} | ConvertTo-Json -Depth 8

$analyzeCall = Try-Call -Name "Analyze capture" -Action {
    Invoke-RestMethod -Method Post -Uri "$BaseUrl$analyzePath" -ContentType "application/json" -Body $analyzeBody
}
if (-not $analyzeCall.ok) {
    Write-Fail "Analyze failed. Stop."
    exit 1
}

$analyze = $analyzeCall.result
if ($analyze.success -eq $true) {
    Write-Pass "Analyze success=true"
} else {
    Write-Fail "Analyze success is not true"
}

$itemCount = @($analyze.items).Count
if ($itemCount -ge 1) {
    Write-Pass "Detected items: $itemCount"
} else {
    Write-Fail "No items detected"
}

$missingSegments = @($analyze.items | Where-Object { -not $_.segmented_png_base64 -or -not $_.raw_crop_base64 }).Count
if ($missingSegments -eq 0) {
    Write-Pass "All detected items include raw + segmented images"
} else {
    Write-Fail "$missingSegments items missing raw/segmented base64"
}

Write-Info "Sample items:"
$analyze.items | Select-Object -First 5 name, category, sub_category, color_code, confidence | Format-Table | Out-String | Write-Host

# 4) Save-selected (first 2)
$selectedIds = @($analyze.items | Select-Object -First 2 | ForEach-Object { $_.item_id })
if (@($selectedIds).Count -eq 0) {
    Write-Fail "No item_id available to save"
    exit 1
}

$saveBody = @{
    user_id = $UserId
    selected_item_ids = $selectedIds
    detected_items = $analyze.items
} | ConvertTo-Json -Depth 50

$saveCall = Try-Call -Name "Save selected items" -Action {
    Invoke-RestMethod -Method Post -Uri "$BaseUrl$savePath" -ContentType "application/json" -Body $saveBody
}

$saved = @()
if ($saveCall.ok) {
    $saved = @($saveCall.result.saved_items)
    $savedCount = $saved.Count
    if ($savedCount -eq @($selectedIds).Count) {
        Write-Pass "Saved item count matches selected count ($savedCount)"
    } else {
        Write-Fail "Saved count mismatch: selected=$(@($selectedIds).Count), saved=$savedCount"
    }

    $invalidUrls = @($saved | Where-Object { -not $_.image_url -or -not $_.raw_image_url }).Count
    if ($invalidUrls -eq 0) {
        Write-Pass "Saved items include R2 URLs"
    } else {
        Write-Fail "$invalidUrls saved items missing image_url/raw_image_url"
    }

    $urlChecks = 0
    $urlFailures = 0
    foreach ($it in ($saved | Select-Object -First 2)) {
        foreach ($u in @($it.image_url, $it.raw_image_url)) {
            if (-not $u) { continue }
            $urlChecks++
            if (Test-UrlReachable -Url $u) {
                Write-Pass "R2 reachable: $u"
            } else {
                $urlFailures++
                Write-Fail "R2 unreachable: $u"
            }
        }
    }
    if ($urlChecks -eq 0) {
        Write-Info "No R2 URLs available for reachability test"
    } elseif ($urlFailures -eq 0) {
        Write-Pass "R2 reachability checks passed ($urlChecks/$urlChecks)"
    } else {
        Write-Fail "R2 reachability failed ($urlFailures/$urlChecks)"
    }

    Write-Info "Saved items summary:"
    $saved | Select-Object item_id, name, category, image_url, raw_image_url | Format-Table | Out-String | Write-Host
}

# 5) Optional board save
$boardDoc = $null
if (-not $SkipBoardSave -and $boardSavePath) {
    $boardBody = @{
        user_id = $UserId
        title = "QA Board $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        occasion = "Occasion"
        description = "Instant QA board upload"
        image_base64 = $imageBase64
        payload = @{
            source = "qa_instant"
            timestamp = (Get-Date).ToString("o")
        }
    } | ConvertTo-Json -Depth 20

    $boardCall = Try-Call -Name "Board save (base64 -> R2 -> saved_boards)" -Action {
        Invoke-RestMethod -Method Post -Uri "$BaseUrl$boardSavePath" -ContentType "application/json" -Body $boardBody
    }

    if ($boardCall.ok) {
        $boardDoc = $boardCall.result.document
        if ($boardDoc -and $boardDoc.imageUrl) {
            Write-Pass "Board saved with imageUrl: $($boardDoc.imageUrl)"
            if (Test-UrlReachable -Url $boardDoc.imageUrl) {
                Write-Pass "Board image R2 reachable"
            } else {
                Write-Fail "Board image R2 unreachable"
            }
        } else {
            Write-Fail "Board saved response missing document.imageUrl"
        }
    }
} elseif ($SkipBoardSave) {
    Write-Info "Skipped board-save test by flag"
}

# 6) Appwrite persistence checks through backend read APIs
$outfitsRead = Try-Call -Name "Appwrite read-back outfits" -Action {
    Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/data/outfits?user_id=$UserId&limit=20"
}
if ($outfitsRead.ok) {
    $docs = @($outfitsRead.result.documents)
    if ($docs.Count -ge 1) {
        Write-Pass "Outfits documents found for user: $($docs.Count)"
    } else {
        Write-Fail "No outfits documents found for user"
    }

    if ($saved.Count -ge 1) {
        $savedImageUrls = @($saved | ForEach-Object { $_.image_url })
        $savedRawUrls = @($saved | ForEach-Object { $_.raw_image_url })
        $matched = @(
            $docs | Where-Object {
                ($savedImageUrls -contains $_.image_url) -or
                ($savedImageUrls -contains $_.masked_url) -or
                ($savedRawUrls -contains $_.image_url) -or
                ($savedRawUrls -contains $_.raw_image_url)
            }
        ).Count
        if ($matched -ge 1) {
            Write-Pass "Saved outfit appears in Appwrite read-back"
        } else {
            Write-Fail "Could not match saved outfit URL in Appwrite read-back"
        }
    }
}

$boardsRead = Try-Call -Name "Appwrite read-back saved_boards" -Action {
    Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/boards?user_id=$UserId&limit=20"
}
if ($boardsRead.ok) {
    $docs = @($boardsRead.result.documents)
    if ($docs.Count -ge 1) {
        Write-Pass "saved_boards documents found for user: $($docs.Count)"
    } else {
        Write-Fail "No saved_boards documents found for user"
    }

    if ($boardDoc -and $boardDoc.imageUrl) {
        $matched = @($docs | Where-Object { $_.imageUrl -eq $boardDoc.imageUrl }).Count
        if ($matched -ge 1) {
            Write-Pass "Saved board appears in Appwrite read-back"
        } else {
            Write-Fail "Could not match saved board in Appwrite read-back"
        }
    }
}

Write-Info "AHVI Instant QA finished"
