# Clinical-Notes-NLP
NLP, intensive modelling of MIMIC-IV-Note (Deidentified free-text clinical notes)

Study:
https://physionet.org/content/mimic-iv-note/2.2/

Documentation:
https://mimic.mit.edu/docs/IV/modules/note/

Refernces:

* https://github.com/MIT-LCP/mimic-code
    * utilized the buildmimic script to create a duckdb database for MIMIC-IV-Note

# Setting Up the MIMIC-IV-NOTE DuckDB Database

From the project root:

```bash
./setup_mimic_note_duckdb.sh
```

When the script is run, it will prompt for the local path to directory containing the MIMIC-IV-NOTE CSV files. (e.g. contains the csv.gz or csv files)

Example:
SHOW TABLES;
```text
/home/username/physionet.org/files/mimic-iv-note/2.2/note
```

The script expects the dataset structure:

```text
note/
├── discharge.csv.gz
├── radiology.csv.gz
└── ...
```

By default, the DuckDB database will be created as:

```text
mimic4_note.db
```

You can optionally specify a custom output path:

```bash
./setup_mimic_note_duckdb.sh data/mimic4_note.db
```

