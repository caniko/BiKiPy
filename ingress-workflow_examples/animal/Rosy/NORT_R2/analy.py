from bikipy.ingress.run import analyze_and_save, generate_inspection_videos, get_ingress

# analyze_and_save(".")
# generate_inspection_videos(["OO670_0"], ".")
get_ingress().purge_cached_reads()
