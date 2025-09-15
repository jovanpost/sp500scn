import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_lake.storage import Storage


def test_list_prefix_normalizes_supabase_items():
    st = Storage()
    st.mode = "supabase"

    class APIResp:
        def __init__(self, data):
            self.data = data

    class Bucket:
        def list(self, prefix):
            assert prefix == "membership"
            return APIResp([{"name": "sp500_members.parquet"}])

    class Client:
        class StorageAPI:
            def __init__(self, bucket):
                self._bucket = bucket

            def from_(self, bucket_name):
                assert bucket_name is st.bucket
                return self._bucket

        def __init__(self, bucket):
            self.storage = Client.StorageAPI(bucket)

    st.bucket = Bucket()
    st.supabase_client = Client(st.bucket)

    items = st.list_prefix("membership/")
    assert items == ["membership/sp500_members.parquet"]
