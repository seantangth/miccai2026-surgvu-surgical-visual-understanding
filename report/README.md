# SurgVU Challenge 2026 — Team Report Template

Thank you for participating in the SurgVU Challenge 2026! Please use this template to submit your final report in LaTeX format. Your report will be included directly in the official challenge publication.

## Getting Started

1. **Fill in your team name, author names, and affiliations in `main.tex`**

   Open `main.tex` and follow the four numbered steps marked with `*** STEP N ***`:

   - **Step 1:** Replace `YOUR-TEAM-NAME` in `\title{}` with your team name (e.g., `ISRG`).
   - **Step 2:** Replace the placeholder `\author{}` lines with your actual author names. Add or remove lines as needed. Example:
     ```latex
     \author[1]{Jane Smith}
     ```
   - **Step 3:** Replace the placeholder `\affil{}` lines with your actual affiliations, numbered to match the authors above. Example:
     ```latex
     \affil[1]{Department of Computer Science, University of Example, City, Country}
     \affil[2]{Institute of Medical Imaging, Another University, City, Country}
     ```
   - **Step 4:** Update `\input{YOUR-TEAM-NAME}` to match your team name (e.g., `\input{ISRG}`).

2. **Rename files and folders to your team name**

   - `YOUR-TEAM-NAME.tex` → `ISRG.tex`
   - `TeamDocs2026/YOUR-TEAM-NAME/` → `TeamDocs2026/ISRG/`

3. **Write your report** in `YOUR-TEAM-NAME.tex` (e.g., `ISRG.tex`) following the section structure and instructions provided inside the file.

4. **Add your references** to `references.bib`. Use the prefix `YOUR-TEAM-NAME_` for all citation keys (e.g., `\cite{ISRG_Smith2024}`).

5. **Place all figures** in the `TeamDocs2026/YOUR-TEAM-NAME/` folder (e.g., `TeamDocs2026/ISRG/`) and reference them as:

   ```latex
   \includegraphics{TeamDocs2026/ISRG/figure1.png}
   ```

   An example figure (`figure1.png`) is already included in the folder.

6. **Compile `main.tex`** to preview your report.

## What to Submit

Please send back the entire report package as a zip file including the following:

1. Your completed `main.tex` (with author names and affiliations filled in)
2. Your completed `YOUR-TEAM-NAME.tex` (e.g., `ISRG.tex`)
3. Your completed `references.bib`
4. All figure files in the `TeamDocs2026/YOUR-TEAM-NAME/` folder (e.g., `TeamDocs2026/ISRG/`)

## File Structure

```
team_report_template/
├── main.tex                    # EDIT THIS: team name, authors, affiliations
├── YOUR-TEAM-NAME.tex          # EDIT THIS: your report content
├── references.bib              # EDIT THIS: your references
├── TeamDocs2026/
│   └── YOUR-TEAM-NAME/         # Your figures go here (e.g., ISRG/)
│       └── figure1.png         # Example figure (replace with your own)
└── README.md                   # This file
```

## Guidelines

- Report length: 1 to 5 pages, word count not to exceed 1000.
- Write in first-person plural voice ("We", "Our", "We propose").
- Do NOT rename or remove section headings (`\subsection`, `\subsubsection`).
- Provide a link to your code repository in the report.
- Disclose any private data or additional annotations used for training.
- Use references to support your claims where possible.
- Please provide sufficient description to re-train your model from scratch, either in this report or in your code repository.

