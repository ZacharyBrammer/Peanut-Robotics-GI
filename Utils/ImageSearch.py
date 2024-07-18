import lancedb


def imageSearch(txt, table_name):
    db = lancedb.connect("embeddings.db")
    table = db.open_table(table_name)
    res = table.search(txt).limit(1).to_pandas()
    return res
