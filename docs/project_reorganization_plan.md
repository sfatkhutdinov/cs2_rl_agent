# CS2 RL Agent Project Reorganization Plan

*Last updated: 2024-03-13 - Initial creation of reorganization plan*

**Tags:** #meta #architecture #organization

## Current Issues

The root directory contains numerous files of different types and purposes:
- Multiple `.bat` files (50+)
- Various training scripts 
- Test scripts
- Configuration files
- Documentation files
- Utility scripts

This creates several problems:
1. Difficult to locate specific files
2. Related files are not grouped together
3. New developers struggle to understand the project structure
4. Increased chance of file conflicts during development
5. Poor separation of concerns

## Proposed Directory Structure

```
cs2_rl_agent/
│
├── README.md                 # Project overview and setup instructions
├── requirements.txt          # Python dependencies
│
├── scripts/                  # Executable scripts
│   ├── training/             # Training script batch files
│   │   ├── train_adaptive.bat
│   │   ├── train_strategic.bat
│   │   └── ...
│   ├── testing/              # Testing script batch files  
│   │   ├── test_cs2_env.bat
│   │   └── ...
│   ├── utils/                # Utility script batch files
│   │   ├── check_gpu.bat
│   │   ├── setup_conda.bat
│   │   └── ...
│   └── deployment/           # Deployment script batch files
│
├── training/                 # Training Python scripts
│   ├── train_adaptive.py
│   ├── train_strategic.py
│   ├── train_discovery.py
│   ├── train_autonomous.py
│   └── ...
│
├── testing/                  # Testing Python scripts
│   ├── test_cs2_env.py
│   ├── test_config.py
│   ├── test_discovery_env.py
│   └── ...
│
├── config/                   # Configuration files
│   ├── adaptive_config.yaml
│   ├── strategic_config.yaml
│   ├── gpu_config.json
│   └── ...
│
├── docs/                     # Documentation (Markdown files)
│   ├── ANACONDA_SETUP.md
│   ├── WINDOWS_SETUP.md
│   ├── ALL_IN_ONE_GUIDE.md
│   ├── AUTONOMOUS_AGENT.md
│   └── ...
│
├── src/                      # Core source code (already organized)
│   ├── agent/
│   ├── actions/
│   ├── environment/
│   ├── utils/
│   ├── interface/
│   └── ...
│
├── analysis_log/             # Analysis documentation (already organized)
│
└── .cursor/                  # Cursor IDE configuration (already organized)
    └── rules/
```

## Migration Plan

1. **Create the new directory structure**
   - Create all necessary directories
   - Keep the existing files in place initially

2. **Move batch files to scripts/ directory**
   - Categorize batch files by purpose
   - Move each file to the appropriate subdirectory
   - Update any internal references

3. **Move Python training scripts to training/ directory**
   - Move all train_*.py files
   - Update imports and relative paths

4. **Move Python test scripts to testing/ directory**
   - Move all test_*.py files
   - Update imports and relative paths

5. **Move documentation to docs/ directory**
   - Move all .md files except README.md and those in analysis_log/
   - Update any cross-references

6. **Update batch files to reference new locations**
   - Modify paths in all batch files to point to the new script locations

7. **Update imports and references in Python files**
   - Ensure all imports and file references work with the new structure

8. **Update documentation to reflect new structure**
   - Update README.md with new directory structure information
   - Update analysis_log/ documents as needed

## Validation Plan

1. **Test each batch file**
   - Ensure each script runs correctly from its new location

2. **Run key training scripts**
   - Verify that training processes work with the new structure

3. **Run key test scripts**
   - Verify that tests execute correctly

4. **Check documentation links**
   - Ensure all documentation references are valid

## Benefits

1. **Improved Navigation**: Files are logically grouped and easier to find
2. **Better Onboarding**: New developers can quickly understand the codebase organization
3. **Reduced Conflicts**: Separate directories reduce the chance of merge conflicts
4. **Clearer Structure**: Clean separation between different types of files
5. **Better Maintainability**: Related files are grouped together
6. **Scalability**: Structure can accommodate future growth 