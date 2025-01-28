SET GLOBAL local_infile = 1;

SHOW VARIABLES LIKE 'local_infile';

SHOW VARIABLES LIKE 'secure_file_priv';


LOAD DATA LOCAL INFILE '/Users/Catherina/Desktop/ds5110/final project/prop_table.csv'
INTO TABLE property
FIELDS TERMINATED BY ','
ENCLOSED BY ''''
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;