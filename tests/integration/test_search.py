"""Integration tests for the fts / vss search helpers."""

from __future__ import annotations

import duckdb
import pytest

import pyduck_ona as pona


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("INSTALL fts; LOAD fts;")
    con.execute("INSTALL vss; LOAD vss;")
    yield con
    con.close()


def test_build_and_drop_fts_index(con):
    con.execute("CREATE TABLE docs(id INTEGER, text VARCHAR);")
    con.execute("INSERT INTO docs VALUES (1, 'hello world'), (2, 'goodbye world');")
    pona.build_fts_index("docs", "id", ["text"], con=con)
    schemas = con.sql("SELECT schema_name FROM information_schema.schemata").fetchall()
    assert any("fts_main_docs" in s for s in schemas)
    pona.drop_fts_index("docs", con=con)
    schemas = con.sql("SELECT schema_name FROM information_schema.schemata").fetchall()
    assert not any("fts_main_docs" in s for s in schemas)


def test_text_search_top_k(con):
    con.execute("""
        CREATE TABLE hris(employee_id VARCHAR, bio VARCHAR);
        INSERT INTO hris VALUES
            ('E1', 'data scientist with python'),
            ('E2', 'hr business partner'),
            ('E3', 'python engineer and manager');
    """)
    rel = pona.text_search(
        "hris",
        "python",
        id_col="employee_id",
        text_cols=["bio"],
        k=2,
        con=con,
    )
    rows = rel.fetchall()
    assert len(rows) == 2
    ids = {r[0] for r in rows}
    assert ids == {"E1", "E3"}


def test_text_search_existing_index(con):
    con.execute("""
        CREATE TABLE hris(employee_id VARCHAR, bio VARCHAR);
        INSERT INTO hris VALUES
            ('E1', 'data scientist with python'),
            ('E2', 'hr business partner');
    """)
    pona.build_fts_index("hris", "employee_id", ["bio"], con=con)
    rel = pona.text_search("hris", "python", id_col="employee_id", con=con)
    rows = rel.fetchall()
    assert any(r[0] == "E1" for r in rows)


def test_build_and_drop_vector_index(con):
    con.execute("""
        CREATE TABLE embeddings(id INTEGER, vec FLOAT[3]);
        INSERT INTO embeddings SELECT i, array_value(random(), random(), random())::FLOAT[3]
        FROM generate_series(1, 50) tbl(i);
    """)
    pona.build_vector_index("embeddings", "vec", metric="l2sq", con=con)
    indexes = con.sql("SELECT index_name FROM duckdb_indexes()").fetchall()
    assert any("embeddings_vec_hnsw_idx" in i for i in indexes)
    pona.drop_vector_index("embeddings", "vec", con=con)
    indexes = con.sql("SELECT index_name FROM duckdb_indexes()").fetchall()
    assert not any("embeddings_vec_hnsw_idx" in i for i in indexes)


def test_vector_search_top_k(con):
    con.execute("""
        CREATE TABLE skills(employee_id VARCHAR, embedding FLOAT[3]);
        INSERT INTO skills VALUES
            ('E1', [1.0, 0.0, 0.0]),
            ('E2', [0.0, 1.0, 0.0]),
            ('E3', [0.9, 0.1, 0.0]);
    """)
    rel = pona.vector_search(
        "skills",
        [1.0, 0.0, 0.0],
        id_col="employee_id",
        vector_col="embedding",
        k=2,
        metric="l2sq",
        con=con,
    )
    rows = rel.fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "E1"
    assert rows[0][1] == pytest.approx(0.0, abs=1e-6)
    assert rows[1][0] == "E3"


def test_vector_search_cosine(con):
    con.execute("""
        CREATE TABLE skills(employee_id VARCHAR, embedding FLOAT[3]);
        INSERT INTO skills VALUES
            ('E1', [1.0, 0.0, 0.0]),
            ('E2', [0.0, 1.0, 0.0]),
            ('E3', [0.9, 0.1, 0.0]);
    """)
    rel = pona.vector_search(
        "skills",
        [1.0, 0.0, 0.0],
        id_col="employee_id",
        vector_col="embedding",
        k=2,
        metric="cosine",
        con=con,
    )
    rows = rel.fetchall()
    assert rows[0][0] == "E1"


def test_fuzzy_join_vectors(con):
    con.execute("""
        CREATE TABLE employees(employee_id VARCHAR, embedding FLOAT[3]);
        INSERT INTO employees VALUES
            ('E1', [1.0, 0.0, 0.0]),
            ('E2', [0.0, 1.0, 0.0]);
        CREATE TABLE roles(role_id VARCHAR, req FLOAT[3]);
        INSERT INTO roles VALUES
            ('R1', [1.0, 0.0, 0.0]),
            ('R2', [0.0, 1.0, 0.0]);
    """)
    rel = pona.fuzzy_join_vectors(
        "employees",
        "roles",
        "embedding",
        "req",
        k=1,
        metric="l2sq",
        con=con,
    )
    rows = rel.fetchall()
    assert len(rows) == 2
    pairs = {(r[0], r[2]) for r in rows}
    assert pairs == {("E1", "R1"), ("E2", "R2")}
