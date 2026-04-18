// Copy nodes.csv, edges.csv, hyperedge_members.csv into Neo4j import directory first.
// Native import: no APOC dependency. Relationship types are expanded explicitly.
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (n:Entity) REQUIRE n.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CALL {
  WITH row
  CREATE (n:Entity {id: row['id:ID']})
  SET n.label = row.label,
      n.file_type = row.file_type,
      n.category = row.category,
      n.source_file = row.source_file
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Document' THEN [1] ELSE [] END | SET n:Document)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Concept' THEN [1] ELSE [] END | SET n:Concept)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Hyperedge' THEN [1] ELSE [] END | SET n:Hyperedge)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'General' THEN [1] ELSE [] END | SET n:General)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'SLAM' THEN [1] ELSE [] END | SET n:SLAM)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Robotics' THEN [1] ELSE [] END | SET n:Robotics)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'SLAM_Supplement' THEN [1] ELSE [] END | SET n:SLAM_Supplement)
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
CALL {
  WITH row
  MATCH (a:Entity {id: row[':START_ID']})
  MATCH (b:Entity {id: row[':END_ID']})
  FOREACH (_ IN CASE WHEN row[':TYPE'] = 'BELONGS_TO' THEN [1] ELSE [] END |
    MERGE (a)-[r:BELONGS_TO]->(b)
    SET r.relation = row.relation,
        r.confidence = row.confidence,
        r.weight = toFloat(row['weight:float']),
        r.source_file = row.source_file
  )
  FOREACH (_ IN CASE WHEN row[':TYPE'] = 'CITES' THEN [1] ELSE [] END |
    MERGE (a)-[r:CITES]->(b)
    SET r.relation = row.relation,
        r.confidence = row.confidence,
        r.weight = toFloat(row['weight:float']),
        r.source_file = row.source_file
  )
  FOREACH (_ IN CASE WHEN row[':TYPE'] = 'CONCEPTUALLY_RELATED_TO' THEN [1] ELSE [] END |
    MERGE (a)-[r:CONCEPTUALLY_RELATED_TO]->(b)
    SET r.relation = row.relation,
        r.confidence = row.confidence,
        r.weight = toFloat(row['weight:float']),
        r.source_file = row.source_file
  )
  FOREACH (_ IN CASE WHEN row[':TYPE'] = 'EVALUATES_WITH' THEN [1] ELSE [] END |
    MERGE (a)-[r:EVALUATES_WITH]->(b)
    SET r.relation = row.relation,
        r.confidence = row.confidence,
        r.weight = toFloat(row['weight:float']),
        r.source_file = row.source_file
  )
  FOREACH (_ IN CASE WHEN row[':TYPE'] = 'RATIONALE_FOR' THEN [1] ELSE [] END |
    MERGE (a)-[r:RATIONALE_FOR]->(b)
    SET r.relation = row.relation,
        r.confidence = row.confidence,
        r.weight = toFloat(row['weight:float']),
        r.source_file = row.source_file
  )
  FOREACH (_ IN CASE WHEN row[':TYPE'] = 'REFERENCES' THEN [1] ELSE [] END |
    MERGE (a)-[r:REFERENCES]->(b)
    SET r.relation = row.relation,
        r.confidence = row.confidence,
        r.weight = toFloat(row['weight:float']),
        r.source_file = row.source_file
  )
  FOREACH (_ IN CASE WHEN row[':TYPE'] = 'SEMANTICALLY_SIMILAR_TO' THEN [1] ELSE [] END |
    MERGE (a)-[r:SEMANTICALLY_SIMILAR_TO]->(b)
    SET r.relation = row.relation,
        r.confidence = row.confidence,
        r.weight = toFloat(row['weight:float']),
        r.source_file = row.source_file
  )
  FOREACH (_ IN CASE WHEN row[':TYPE'] = 'SHARES_DATA_WITH' THEN [1] ELSE [] END |
    MERGE (a)-[r:SHARES_DATA_WITH]->(b)
    SET r.relation = row.relation,
        r.confidence = row.confidence,
        r.weight = toFloat(row['weight:float']),
        r.source_file = row.source_file
  )
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///hyperedge_members.csv' AS row
CALL {
  WITH row
  MATCH (a:Entity {id: row[':START_ID']})
  MATCH (b:Entity {id: row[':END_ID']})
  MERGE (a)-[r:HAS_MEMBER]->(b)
  SET r.relation = row.relation,
      r.confidence = row.confidence,
      r.weight = toFloat(row['weight:float']),
      r.source_file = row.source_file
} IN TRANSACTIONS OF 1000 ROWS;
