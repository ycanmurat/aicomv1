from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class KnowledgeHit:
    title: str
    body: str
    source: str
    score: float


class KnowledgeStore:
    """Ağ gerektirmeyen küçük bilgi tabanı; SQLite FTS5 ile Türkçe arama yapar."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'local',
                    created_at TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    title,
                    body,
                    content='knowledge_documents',
                    content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );
                CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge_documents BEGIN
                    INSERT INTO knowledge_fts(rowid, title, body)
                    VALUES (new.id, new.title, new.body);
                END;
                CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge_documents BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, title, body)
                    VALUES ('delete', old.id, old.title, old.body);
                END;
                CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge_documents BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, title, body)
                    VALUES ('delete', old.id, old.title, old.body);
                    INSERT INTO knowledge_fts(rowid, title, body)
                    VALUES (new.id, new.title, new.body);
                END;
                """
            )

    def add(self, *, title: str, body: str, source: str = "local") -> int:
        clean_title = " ".join(title.split()).strip()
        clean_body = " ".join(body.split()).strip()
        clean_source = " ".join(source.split()).strip() or "local"
        if not clean_title or not clean_body:
            raise ValueError("Bilgi başlığı ve içeriği boş olamaz.")
        if len(clean_body) > 100_000:
            raise ValueError("Tek bilgi belgesi en fazla 100.000 karakter olabilir.")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO knowledge_documents(title, body, source, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean_title, clean_body, clean_source, datetime.now(UTC).isoformat()),
            )
            return int(cursor.lastrowid)

    def search(self, query: str, *, limit: int = 3) -> list[KnowledgeHit]:
        tokens = [token for token in _TOKEN.findall(query.lower()) if len(token) > 2]
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:12])
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.title, d.body, d.source, bm25(knowledge_fts, 4.0, 1.0) AS score
                FROM knowledge_fts
                JOIN knowledge_documents d ON d.id = knowledge_fts.rowid
                WHERE knowledge_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, max(1, min(limit, 8))),
            ).fetchall()
        return [
            KnowledgeHit(
                title=str(row["title"]),
                body=str(row["body"]),
                source=str(row["source"]),
                score=float(row["score"]),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM knowledge_documents").fetchone()
        return int(row["count"])
