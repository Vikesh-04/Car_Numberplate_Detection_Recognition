# Car_Numberplate_Detection_Recognition
Flask application that detects the car number plate using YOLOv3 and performs number recognition using OCR.space API and validates if car entry exist in PostgreSQL database.<br />

YOLOv3  model was trained on 1000+ car images <br />

Prerequisite - <br />
- YOLOv3 model .weights file <br />
- YOLOv3 model testing .cfg file <br />
- OCR.space API key for number recognition <br />
- PostgreSQL database
- psycopg2 library for connecting to PostgreSQL database
- flask for deployment

