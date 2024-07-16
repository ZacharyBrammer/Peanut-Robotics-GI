import lancedb

def imageSearch(txt, table_name):
    db = lancedb.connect("embeddings.db")
    print(db.table_names())
    print(table_name)
    table = db.open_table(table_name)
    res = table.search(txt).limit(1).to_pandas()
    return res
