from backend_module.database import DataBaseManager

def queue_unencoded_video_jobs(db_manager: DataBaseManager):
    db_manager.query(
        "INSERT INTO encode_job (SELECT time::now() AS created_at, id AS file, 'queued' AS status FROM file WHERE encode = 'video-none' AND mime ~ 'video/' AND id NOTINSIDE (SELECT VALUE file FROM encode_job));"
    )

def get_queued_job(db_manager: DataBaseManager):
    res = db_manager.query(
        """
        SELECT 
            *
        FROM 
            encode_job
        WHERE 
            status = 'queued'
        ORDER 
            BY created_at ASC
        LIMIT 
            math::max([
                $limit_in_progress - array::len((SELECT id FROM encode_job WHERE status = 'in_progress')),
                0
            ]);
        """,
        {"limit_in_progress": 3},
    )
    print(res[0]["file"])
    return res