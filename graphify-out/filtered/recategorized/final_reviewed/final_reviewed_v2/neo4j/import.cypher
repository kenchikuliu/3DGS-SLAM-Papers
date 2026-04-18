// Copy nodes.csv, edges.csv, hyperedge_members.csv into Neo4j import directory first.
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
  CALL apoc.create.relationship(
    a,
    row[':TYPE'],
    {
      relation: row.relation,
      confidence: row.confidence,
      weight: toFloat(row['weight:float']),
      source_file: row.source_file
    },
    b
  ) YIELD rel
  RETURN rel
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
