from stufio.core.migrations.base import ClickhouseMigrationScript
from stufio.db.clickhouse import get_database_from_dsn

class CreateQueryProfilesTable(ClickhouseMigrationScript):
    name = "create_query_profiles_table"
    description = "Create table for storing detailed query profiling data"
    migration_type = "schema"
    order = 10

    async def run(self, db) -> None:
        db_name = get_database_from_dsn()

        await db.command(
            f"""
            CREATE TABLE IF NOT EXISTS `{db_name}`.`query_profiles` (
                id UUID DEFAULT generateUUIDv4(),
                correlation_id String,
                session_id String DEFAULT '',
                request_path String DEFAULT '',
                database_type String,
                query String,
                operation_type String,
                duration_ms Float64,
                timestamp DateTime DEFAULT now(),  -- Changed from DateTime64(3) to regular DateTime
                parameters String DEFAULT '',
                collection_or_table String DEFAULT '',
                is_slow UInt8 DEFAULT 0,
                source_module String DEFAULT '',
                source_function String DEFAULT '',
                stacktrace String DEFAULT '',
                result_size Int32 DEFAULT 0,
                status String DEFAULT 'success',
                error_message String DEFAULT '',
                app_name String DEFAULT '',
                user_id String DEFAULT ''
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMMDD(timestamp)
            ORDER BY (timestamp, correlation_id, database_type)
            TTL timestamp + INTERVAL 7 DAY DELETE  -- Now using regular DateTime in TTL
            SETTINGS index_granularity = 8192;
            """
        )

        # Create an optimized summary table that will be populated with a materialized view
        await db.command(
            f"""
            CREATE TABLE IF NOT EXISTS `{db_name}`.`query_profiles_summary` (
                date Date,
                correlation_id String,
                database_type String,
                request_path String,
                session_id String,
                query_count UInt64,
                total_duration_ms Float64,
                avg_duration_ms Float64,
                max_duration_ms Float64,
                slow_query_count UInt64,
                error_count UInt64
            ) ENGINE = SummingMergeTree()
            PARTITION BY toYYYYMM(date)
            ORDER BY (date, correlation_id, database_type, request_path)
            TTL date + INTERVAL 30 DAY DELETE
            SETTINGS index_granularity = 8192;
            """
        )