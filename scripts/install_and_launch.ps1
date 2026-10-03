param()

$adb = ".\android_tools\platform-tools\adb.exe"

Write-Host "Waiting for device authorization on phone screen..." -ForegroundColor Cyan
& $adb wait-for-device

Write-Host "Device authorized! Installing Spatial App (com.spatial.ai.app)..." -ForegroundColor Yellow
$installResult = & $adb install -r -d "web_ui\static\downloads\spatial_app.apk"
Write-Host $installResult -ForegroundColor Green

Write-Host "Configuring USB tunnel (tcp:5000)..." -ForegroundColor Cyan
& $adb reverse tcp:5000 tcp:5000

Write-Host "Launching Spatial App on your phone..." -ForegroundColor Magenta
& $adb shell monkey -p com.spatial.ai.app -c android.intent.category.LAUNCHER 1

Write-Host "`nSUCCESS! Spatial App is now installed and open on your device!" -ForegroundColor Green
