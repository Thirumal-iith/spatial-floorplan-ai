# scripts/connect_android_usb.ps1
# Automatically forwards the Spatial AI port over USB cable to your Android phone
# and launches the app directly on the phone screen!

$ADB = Join-Path $PSScriptRoot "..\android_tools\platform-tools\adb.exe"

if (-not (Test-Path $ADB)) {
    Write-Host "[ERROR] adb.exe not found at $ADB" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   SPATIAL AI - ANDROID USB CABLE DIRECT CONNECTION    " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Connect your Android phone to this PC via USB cable." -ForegroundColor Yellow
Write-Host "2. Make sure USB Debugging is ON in Settings -> Developer Options." -ForegroundColor Yellow
Write-Host "3. If your phone asks to Allow USB debugging, tap Allow / OK." -ForegroundColor Yellow
Write-Host ""

Write-Host "Waiting for Android device..." -ForegroundColor Gray

while ($true) {
    $devices = & $ADB devices | Select-String -Pattern "\tdevice$"
    if ($devices) {
        $devId = ($devices[0] -split "\t")[0]
        Write-Host "[SUCCESS] Android device detected: $devId" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "Mapping port 5000 over USB cable..." -ForegroundColor Cyan
& $ADB reverse tcp:5000 tcp:5000

Write-Host ""
Write-Host "Launching Spatial AI app on your phone..." -ForegroundColor Cyan
& $ADB shell am start -a android.intent.action.VIEW -d "http://localhost:5000"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "   CONNECTED! The app is now open on your Android phone! " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host "To install permanently onto your home screen:" -ForegroundColor White
Write-Host "  1. On your phone screen in Chrome, tap the 3-dots menu." -ForegroundColor Gray
Write-Host "  2. Tap Install app (or Add to Home screen)." -ForegroundColor Gray
Write-Host "  3. The native app icon will be installed into your phone app drawer!" -ForegroundColor Gray
Write-Host ""
