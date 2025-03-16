import os
import datetime
import shutil

def backup_database(db_path, backup_dir):
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    db_name = os.path.basename(db_path)
    backup_name = f"{os.path.splitext(db_name)[0]}_{timestamp}.db"
    backup_path = os.path.join(backup_dir, backup_name)
    
    shutil.copy2(db_path, backup_path)
    print(f"Backup created: {backup_path}")

if __name__ == "__main__":
    DB_PATH = "path/to/sqlite.db"
    BACKUP_DIR = "path/to/backup/directory"
    backup_database(DB_PATH, BACKUP_DIR)