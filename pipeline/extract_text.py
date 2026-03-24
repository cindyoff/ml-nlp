import sqlite3
from .config import Path_Sciencespo
from pathlib import Path
from tqdm import tqdm

def index_database(db_path: Path, vacuum: bool = False):
    """
    Add performance indexes to an Arkindex SQLite export.
    """
    from arkindex_export import open_database, database

    # Initialize database connection
    open_database(db_path)

    if database.is_closed():
        database.connect()

    with database.atomic():
        # Critical for recursive CTE performance
        database.execute_sql("""
            CREATE INDEX IF NOT EXISTS idx_elementpath_parent_child
            ON element_path(parent_id, child_id);
        """)

        database.execute_sql("""
            CREATE INDEX IF NOT EXISTS idx_elementpath_child
            ON element_path(child_id);
        """)

        # Useful if filtering on Element.type
        database.execute_sql("""
            CREATE INDEX IF NOT EXISTS idx_element_type
            ON element(type);
        """)

    if vacuum:
        print("Running VACUUM (this may take time)...")
        database.execute_sql("VACUUM;")

    database.close()

    print("Indexing completed.")




def extract_to_txt():
    from arkindex_export import open_database, Element, Transcription
    from arkindex_export.queries import list_children
    TEXT_FOLDER = "data/text_files"
    path_txt = Path(TEXT_FOLDER)
    path_txt.mkdir(exist_ok=True)
    YEARS = ['1981', '1988', '1993']
    ELECTIONS = ['legislatives', 'presidentielle']
    folder_id = {}
    for year in YEARS:
        folder_id[year] = {}
        # ❌ Supprimez la création de dossiers ici

    # Assignez d'abord les IDs
    folder_id['1981']['legislatives'] = 'd51ea3db-68ee-4cc0-a87f-736ee17c5f87'
    folder_id['1988']['legislatives'] = 'dfba9f5c-02de-478c-85c5-0ee780455433'
    folder_id['1993']['legislatives'] = 'cf29300f-40bf-4b61-be93-6cb631be8fab'
    folder_id['1988']['presidentielle'] = 'fd5bee0a-83e8-4bdc-aa48-52331af2e151'

    # Créez uniquement les dossiers qui ont un ID
    for year, types in folder_id.items():
        for type, fid in types.items():
            if fid:
                type_folder = path_txt / year / type
                type_folder.mkdir(parents=True, exist_ok=True)

    # compute some statistics
    print("Number of folders", Element.select().where(Element.type == 'folder').count())
    print("Number of pages:", Element.select().where(Element.type == 'page').count())

    for year in YEARS:
        print ('year', year)
        for e_type in ELECTIONS:
            print ('elections', e_type)
            f_id = folder_id[year].get(e_type, None)
            if f_id:
                documents = list_children(f_id).where(Element.type == 'document')
                print(f_id,"Number of documents", documents.count())
                transcriptions_number = 0
                for document in tqdm(documents):
                    pages = list_children(document.id).where(Element.type == 'page')
                    transcriptions = ""
                    for page in pages:
                        page_transcription = Transcription.select().where(Transcription.element == page.id).first()
                        if page_transcription:
                            transcriptions += page_transcription.text

                    if transcriptions:
                        with open(f"{TEXT_FOLDER}/{year}/{e_type}/{document.name}.txt", "w") as f:
                            f.write(transcriptions)
                        transcriptions_number += 1
                print("Number of transcriptions", transcriptions_number)

if __name__ == "__main__":
    from arkindex_export import open_database
    # Index the database before opening it
    index_database(Path_Sciencespo)

    # load the  export
    open_database(Path_Sciencespo)

    # create a folder to store the text files
    extract_to_txt()
