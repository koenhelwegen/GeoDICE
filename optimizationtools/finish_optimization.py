import lib.dp.coordinator as coordinator
import logging

logging.basicConfig(level=logging.DEBUG, filename="temp.log", filemode="w")


dpc = coordinator.load(filename="./results/dpc_backup.pickle")

dpc.fit()

dpc.save()
