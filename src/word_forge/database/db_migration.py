"""Database schema migration utilities."""

import sqlite3
from typing import Dict, Optional

from word_forge.database.db_manager import DBManager


class SchemaMigrator:
    """Handles database schema migrations to ensure compatibility across versions."""

    def __init__(self, db_manager: DBManager):
        """Initialize with reference to a DB manager."""
        self.db_manager = db_manager

    def migrate_all(self) -> Dict[str, bool]:
        """Run all required migrations and return results."""
        results = {}

        results["relationship_types_last_updated"] = (
            self.add_last_updated_to_relationship_types()
        )

        return results

    def add_last_updated_to_relationship_types(self) -> bool:
        """Add last_updated column to relationship_types table if needed."""
        try:
            with self.db_manager._create_connection() as conn:
                cursor = conn.cursor()

                # Check if column exists
                cursor.execute("PRAGMA table_info(relationship_types)")
                columns = [info[1] for info in cursor.fetchall()]

                if "last_updated" not in columns:
                    cursor.execute(
                        "ALTER TABLE relationship_types ADD COLUMN last_updated REAL NOT NULL DEFAULT 0"
                    )
                    conn.commit()
                    return True

                return False
        except sqlite3.Error:
            return False


def run_migrations(db_path: Optional[str] = None) -> Dict[str, int]:
    """Run all database migrations on the specified database."""
    db_manager = DBManager(db_path)
    migrator = SchemaMigrator(db_manager)
    return migrator.migrate_all()


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    changes = run_migrations(db_path)

    print("Word Forge Database Migration")
    print("=============================")
    for name, count in changes.items():
        status = "✓" if count > 0 else "-"
        print(f"{status} {name}: {count} changes")
