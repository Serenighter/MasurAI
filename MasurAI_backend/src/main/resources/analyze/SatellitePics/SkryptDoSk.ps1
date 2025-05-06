Get-ChildItem -File | ForEach-Object {
    $oldName = $_.Name
    $extension = $_.Extension
    $newNameBase = $oldName.Substring(0, [Math]::Min(10, $oldName.Length - $extension.Length))
    $newName = $newNameBase + $extension
    Rename-Item -Path $_.FullName -NewName $newName
}
