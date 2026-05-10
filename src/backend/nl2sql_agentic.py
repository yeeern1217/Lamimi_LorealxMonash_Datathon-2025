import os
import time
import re
import logging
from typing import Tuple
import pandas as pd
import psycopg2
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Groq Setup ---
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# --- Supabase Setup ---
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# --- Postgres Direct Connection ---
try:
    pg_conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        dbname=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        port=os.getenv("PG_PORT", 5432)
    )
    pg_conn.autocommit = True
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    raise

def get_schema_description():
    """Fetch database schema with table columns, relationships, and table descriptions."""
    columns_query = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name, ordinal_position;
    """

    relationships_query = """
    SELECT
        tc.table_name AS source_table,
        kcu.column_name AS source_column,
        ccu.table_name AS target_table,
        ccu.column_name AS target_column
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name
    JOIN information_schema.constraint_column_usage ccu
      ON ccu.constraint_name = tc.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY';
    """

    descriptions_query = """
    SELECT
        c.relname AS table_name,
        obj_description(c.oid) AS table_description
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relkind = 'r'  -- only regular tables
      AND n.nspname = 'public';
    """

    schema = {}
    with pg_conn.cursor() as cur:
        # Columns
        cur.execute(columns_query)
        for table, column, dtype in cur.fetchall():
            schema.setdefault(table, {"columns": [], "relationships": [], "description": None})
            schema[table]["columns"].append(f"{column} ({dtype})")

        # Relationships
        cur.execute(relationships_query)
        for source_table, source_column, target_table, target_column in cur.fetchall():
            if source_table in schema:
                schema[source_table]["relationships"].append(
                    f"FOREIGN KEY ({source_column}) REFERENCES {target_table}({target_column})"
                )

        cur.execute(descriptions_query)
        for table, desc in cur.fetchall():
            if table in schema:
                schema[table]["description"] = desc

    return schema


def validate_sql_query(sql: str) -> Tuple[bool, str]:
    """Validate SQL to prevent harmful queries and injections."""
    sql_upper = sql.upper().strip()
    dangerous_ops = [
        r"\bDROP\b", r"\bDELETE\b", r"\bTRUNCATE\b", r"\bALTER\b",
        r"\bCREATE\b", r"\bINSERT\b", r"\bUPDATE\b", r"\bGRANT\b",
        r"\bREVOKE\b", r"\bEXECUTE\b", r"\bSHUTDOWN\b", r"\bKILL\b"
    ]
    for pattern in dangerous_ops:
        if re.search(pattern, sql_upper):
            return False, f"Dangerous operation detected: {pattern}"

    if not sql_upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed"

    injection_patterns = [
        r"UNION\s+ALL",   
        r"1\s*=\s*1",    
        r"OR\s+1\s*=\s*1" 
    ]
    for pattern in injection_patterns:
        if re.search(pattern, sql_upper, re.IGNORECASE):
            return False, "Potential SQL injection detected"
    return True, "Valid SQL query"

def generate_sql_query(user_query: str, schema: dict) -> str:
    """Use Groq LLM to generate SQL query from natural language."""
    schema_text = "\n".join([
        f"Table: {table}\n  Columns: {', '.join(data['columns'])}\n  Relationships: {', '.join(data['relationships']) if data['relationships'] else 'None'}"
        for table, data in schema.items()
    ])

    system_prompt = """You are an expert SQL query generator specializing in PostgreSQL. Your role is to convert natural language questions into precise, efficient SQL queries.
    CRITICAL RULES:
    1. ONLY generate SELECT queries - no data modification operations
    2. Use proper JOIN syntax based on foreign key relationships
    3. Always include WHERE clauses when filtering is implied
    4. Use appropriate aggregations (COUNT, SUM, AVG, MAX, MIN) when needed
    5. Include GROUP BY when using aggregate functions
    6. Use ORDER BY and LIMIT for top-N queries
    7. Apply proper table aliases for readability
    8. Handle date/time functions appropriately
    9. Refer to the table descriptions for more context about the attributes in the data.
    10. Return ONLY the SQL query with no further explanations"""

    user_prompt = f"""Database Schema:
    {schema_text}

    Current User Question: "{user_query}"

    Generate a PostgreSQL query that answers this question accurately. Consider the schema relationships.

    SQL Query:"""


    response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

    sql_query = response.choices[0].message.content.strip()
    sql_query = re.sub(r'```sql|```', '', sql_query).strip()
    is_valid , msg = validate_sql_query(sql_query)
    if not is_valid:
        raise ValueError(f"Generated SQL is invalid: {msg}")
    return sql_query

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('PG_USER')}:{quote_plus(os.getenv('PG_PASSWORD'))}"
    f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT', 5432)}/{os.getenv('PG_DATABASE')}"
)

def execute_sql(sql: str) -> pd.DataFrame:
    """Execute SQL safely and return results as DataFrame."""
    try:
        df = pd.read_sql(sql, engine)
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def response_chat(user_query: str, df: pd.DataFrame, sql_query: str = "") -> str:
    """Use Groq to turn SQL results into natural language answer with insights."""
    if "error" in df.columns:
        return f"I encountered an error with query: {sql_query}\nError: {df['error'][0]}"

    if df.empty:
        return "The query returned no results for your request."

    data_preview = f"""
    Query executed: {sql_query}
    Rows returned: {len(df)}
    Preview:
    {df.head().to_string(index=False)}
    """

    system_prompt = """You are a helpful data assistant. 
    Your job is to answer the user's question directly based only on the SQL results provided. 
    Keep the response simple, clear, and conversational. 
    Do not mention SQL queries, databases, or technical details. 
    Do not add extra reasoning or caveats beyond what was asked. 
    If there are no results, just say so plainly."""

    user_prompt = f"""User Question: "{user_query}"

    Query Results:
    {data_preview}

    Provide a clear and concise summary with insights:"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I found {len(df)} records. Here's a preview:\n{df.head().to_string(index=False)}"

def chat_with_db(user_query: str) -> str:
    """Main NL2SQL pipeline."""

    try:
        # Get schema
        schema = get_schema_description()
        if not schema:
            return "I'm unable to access the database schema. Please check the database connection."
        sql_query = generate_sql_query(user_query, schema)
        
        logger.info(f"Generated SQL: {sql_query}")
        df = execute_sql(sql_query)
        response = response_chat(user_query, df, sql_query)        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat pipeline: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again."