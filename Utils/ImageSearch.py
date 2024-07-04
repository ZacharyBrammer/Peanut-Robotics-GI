import TextConvert
# import ImageConvert
#import os 
import lancedb

# table = ImageConvert.process_images("40777060/40777060_frames/lowres_wide/", "40777060/40777060_frames/lowres_wide.traj")

db = lancedb.connect("embeddings.db/image_embeddings.lance")

print(db.table_names())

table = db.open_table("data")


res = table.search(TextConvert.embed_txt("chair")).limit(5).to_pandas()

print(res)