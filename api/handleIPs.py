from datetime import datetime


def blacklistIP(mongo, ip):
    return mongo.db.clientips.update_one(
        {"clientIP": ip},
        {"$set": {"blacklisted": True, "blacklistTime": datetime.utcnow()}}
    )
