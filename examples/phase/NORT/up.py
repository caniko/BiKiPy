from bikipy.cli.ingress import _define_project_root_directory
from bikipy.ingress.workflow.core import auto_define_ingress_object

# init_settings("phase", "nort", ".", ".parquet")
auto_define_ingress_object(_define_project_root_directory(".")).update_settings(delete_outdated=False)
