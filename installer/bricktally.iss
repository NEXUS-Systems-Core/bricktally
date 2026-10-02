; Per-user install. The uninstaller stays outside {app} so the in-app
; folder swap can replace Programs\BrickTally without removing unins000.exe.
; Do not move UninstallFilesDir into {app}.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{B40A527C-5D9E-40AF-9B5C-42BA8EE6974B}}
AppName=BrickTally
AppVersion={#AppVersion}
AppPublisher=NEXUS Systems
DefaultDirName={localappdata}\Programs\BrickTally
DefaultGroupName=BrickTally
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableWelcomePage=yes
DisableReadyPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=BrickTallySetup
SetupIconFile=bricktally.ico
UninstallDisplayIcon={app}\BrickTally.exe
UninstallFilesDir={localappdata}\Programs\BrickTally-uninstall
CloseApplications=yes
CloseApplicationsFilter=BrickTally.exe,BrickTallyUpdater.exe
RestartApplications=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
VersionInfoVersion={#AppVersion}.0
VersionInfoProductName=BrickTally
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
UsePreviousAppDir=yes

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: checkedonce

[Files]
Source: "..\dist\BrickTally\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\BrickTally"; Filename: "{app}\BrickTally.exe"
Name: "{autodesktop}\BrickTally"; Filename: "{app}\BrickTally.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\BrickTally.exe"; Description: "Launch BrickTally"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{app}.prev"
Type: filesandordirs; Name: "{app}.incoming"
