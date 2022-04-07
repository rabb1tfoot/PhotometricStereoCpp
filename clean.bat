::del /s *.suo;*.sdf;*.ipch;*.obj;*.exp;*.log;*.ilk;*.pch;*.pdb;*.idb;*.tlog;*.lastbuildstate;*.tlh;*.tli;*.aps
::del /s /a:h *.suo;*.sdf;*.ipch;*.obj;*.exp;*.log;*.ilk;*.pch;*.pdb;*.idb;*.tlog;*.lastbuildstate;*.tlh;*.tli;*.aps
:: CUDA는 suo 파일 지우면 컴파일 에러뜸
del /s *.sdf;*.ipch;*.obj;*.exp;*.log;*.ilk;*.pch;*.pdb;*.idb;*.tlog;*.lastbuildstate;*.tlh;*.tli;*.aps
del /s /a:h *.sdf;*.ipch;*.obj;*.exp;*.log;*.ilk;*.pch;*.pdb;*.idb;*.tlog;*.lastbuildstate;*.tlh;*.tli;*.aps
