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

    st.bucket = Bucket()
    items = st.list_prefix("membership/")
    assert items == ["membership/sp500_members.parquet"]
