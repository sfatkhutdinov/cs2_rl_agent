# CS2 RL Agent Documentation - Cursor Project Rules

*Last updated: 2024-03-13 - Updated Current Work section with completed documentation tasks*

**Tags:** #meta #documentation #guidelines #process

## Documentation Structure
- Maintain the established hierarchical organization with `main.md` as the central index
- Keep related documents in appropriate subdirectories (architecture/, components/, testing/, etc.)
- Ensure all documents are properly linked from their parent/index documents

## Version Tracking
- Include a "Last updated" note at the top of each document after the title when making significant changes
- Format: `*Last updated: YYYY-MM-DD - Brief description of changes*`
- No separate maintenance log - change information stays with the relevant document

## Tagging System
- All documents must include a tags section immediately after the title
- Format: `**Tags:** #tag1 #tag2 #tag3`
- Only use tags defined in `document_tags.md`
- Add new tags to `document_tags.md` when necessary

## Document Organization
- Each analysis document should follow the standard structure:
  1. Context/Introduction
  2. Methodology
  3. Findings
  4. Relationship to Other Components
  5. Optimization Opportunities
  6. Next Steps/Recommendations
- Include backlinks to related documents where appropriate
- Use consistent section headings across similar document types

## Navigation and References
- Verify all document links before committing changes
- Provide backlinks to parent/index documents
- Update the "Completed Analyses" section in `main.md` when new analyses are added
- Use the checkmark (✓) to indicate completed analyses

## Formatting Conventions
- Use H1 (#) for document titles only
- Use H2 (##) for major sections
- Use H3 (###) and below for subsections
- Use bullet points for lists of items
- Use numbered lists for sequential steps or prioritized items
- Use code blocks with appropriate language specification for code snippets

## Summary Documents
- For complex topics, create both a detailed analysis document and a summary document
- Summary documents should be concise and focus on key takeaways
- Name summary documents with a clear suffix (e.g., `_summary.md`)

## Testing Documentation
- Maintain separate section files for distinct testing aspects
- Keep the testing_infrastructure.md file as an index to testing documentation
- Ensure testing documentation includes concrete examples and expected outcomes

## Images and Diagrams
- Store all visuals in the visualization/ directory
- Include descriptive alt text for all images
- Reference the source or generation method for all diagrams

## AI Context Management
- At the beginning of each session, review main.md and cursor_project_rules.md to refresh context
- When implementing changes, immediately update relevant documents with version notes
- When creating new documents, immediately add references in the appropriate index files
- For each major documentation update, ensure there's a record in the relevant document's version history
- Keep track of in-progress work in the "Current Work" section below

## Document Update Requests
- Prioritize documentation updates based on user instructions
- When a documentation change is requested, first understand if it affects:
  1. The overall documentation structure (highest priority)
  2. Key index documents (high priority)
  3. Individual analysis documents (normal priority)
- Before implementing changes, confirm the full scope of the request
- After implementing changes, verify all affected references and links

## Progress Tracking
- Maintain a "Current Work" section in this document to track ongoing documentation efforts
- Update the "Completed Analyses" section in main.md when new analyses are completed
- When undertaking multi-step documentation tasks, outline the full plan before starting
- Track dependencies between documents to ensure consistent updates

## Maintenance Guidelines
- Regularly verify link integrity using the link checker tool
- When restructuring documentation, update all affected reference documents
- Never create growing logs that would consume context window space
- Prefer in-place documentation updates with version notes over separate logs 

## Current Work
*Current documentation tasks in progress:*
- Initial setup of project documentation structure ✓
- Creation of cursor_project_rules.md ✓
- Integration of last updated notes in key documents ✓
- Removal of separate maintenance log in favor of in-document version notes ✓
- Created training directory and initial training script analyses ✓
- Updated document_tags.md with new tags for training documentation ✓

*Upcoming documentation tasks:*
1. Training Scripts Analysis
   - Analyze train_adaptive.py and document the adaptive training approach ✓
   - Analyze train_strategic.py and document the strategic agent training ✓
   - Create a comparative overview of different training approaches ✓
   - Analyze other training scripts (train_autonomous.py, train_discovery.py)

2. Agent Implementations Analysis
   - Document the adaptive_agent.py implementation in depth
   - Document the strategic_agent.py implementation
   - Analyze agent mode-switching mechanisms

3. Environment and Configuration Analysis
   - Document the environment implementations in src/environment/
   - Analyze the configuration system and settings

4. Batch Scripts Organization
   - Categorize and document the various .bat scripts and their purposes
   - Create a batch scripts reference document

5. Testing Scripts Analysis
   - Document the test scripts not yet covered in testing documentation
   - Link test scripts to the existing testing documentation 