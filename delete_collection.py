
def delete_collection(db, collection_name):
    db.delete_collection(collection_name)
    db.persist()

