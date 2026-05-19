Activate the DuckDB CLI to interact with the `mimic4_note.db` database:

```bash
duckdb mimic4_note.db
```

To list the tables in the database:

```sql
SET schema 'mimiciv_note';

SHOW TABLES;

-- Alternative (works from main and specific schema):
SHOW ALL TABLES;
```

To inspect the columns of a table:

```sql
DESCRIBE discharge;
```

To preview rows from a table:

```sql
SELECT *
FROM discharge
LIMIT 5;
```

To count the number of rows in a table:

```sql
SELECT COUNT(*)
FROM discharge;
```

To return to the default schema:

```sql
SET schema 'main';
```