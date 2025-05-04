from stufio.core.migrations.base import ClickhouseMigrationScript
from stufio.db.clickhouse import get_database_from_dsn

class CreateQueryProfilesMaterializedView(ClickhouseMigrationScript):
    name = "create_query_profiles_materialized_view"
    description = "Create materialized view to populate query profiles summary table"
    migration_type = "schema"
    order = 20  # Run after the main table creation

    async def run(self, db) -> None:
        db_name = get_database_from_dsn()

        # First check if the table exists
        result = await db.command(
            f"""
            SELECT count() FROM system.tables 
            WHERE database = '{db_name}' AND name = 'query_profiles';
            """
        )
        
        # Skip if the source table doesn't exist
        if not result or result == 0:
            print(f"Table {db_name}.query_profiles does not exist, skipping materialized view creation")
            return

        # Drop the view if it exists to ensure we can recreate it with updated schema
        await db.command(
            f"""
            DROP VIEW IF EXISTS `{db_name}`.`query_profiles_mv`;
            """
        )

        # Create materialized view to automatically populate the summary table
        await db.command(
            f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS `{db_name}`.`query_profiles_mv`
            TO `{db_name}`.`query_profiles_summary`
            AS SELECT 
                toDate(timestamp) AS date,
                correlation_id,
                database_type,
                request_path,
                session_id,
                count() AS query_count,
                sum(duration_ms) AS total_duration_ms,
                avg(duration_ms) AS avg_duration_ms,
                max(duration_ms) AS max_duration_ms,
                countIf(is_slow = 1) AS slow_query_count,
                countIf(status != 'success') AS error_count
            FROM `{db_name}`.`query_profiles`
            GROUP BY date, correlation_id, database_type, request_path, session_id;
            """
        )

        # Create an index on correlation_id for faster lookups
        await db.command(
            f"""
            ALTER TABLE `{db_name}`.`query_profiles` 
            ADD INDEX IF NOT EXISTS idx_correlation_id (correlation_id) TYPE minmax;
            """
        )

        # Create an index on session_id for faster lookups
        await db.command(
            f"""
            ALTER TABLE `{db_name}`.`query_profiles` 
            ADD INDEX IF NOT EXISTS idx_session_id (session_id) TYPE minmax;
            """
        )