                          USCRN/USRCRN DAILY FILES

                            UPDATED: 2017-07-06 
                 
README CONTENTS:                          
    1. GENERAL INFORMATION 
    2. DATA VERSION / STATUS UPDATES
    3. DIRECTORY / FILE ORGANIZATION
    4. FILE AND FILENAME FORMATS
        A. YEARLY
        B. SNAPSHOTS
        C. UPDATES
    5. DATA FIELDS / FORMATS / IMPORTANT NOTES

********************************************************************************

1. GENERAL INFORMATION

NCDC provides access to daily data from the U.S. Climate Reference Network / 
U.S. Regional Climate Reference Network (USCRN/USRCRN) via anonymous ftp at:

        ftp://ftp.ncdc.noaa.gov/pub/data/uscrn/products/daily01
        
and an identical web interface at:

        http://www1.ncdc.noaa.gov/pub/data/uscrn/products/daily01
        
Before using these data, be sure to review this document carefully, as well
as any announcements within the main (daily01) directory. 

********************************************************************************

2. DATA VERSION / STATUS UPDATES

Status and Version Information for U.S. Climate Reference Network Data

  ##########################
  
Version change:     2.1 to 2.1.1
Commencement Date:  2017-06-20
Completion Date:    2017-06-30
Variables impacted: Precipitation

    Effective June 30, 2017, recalculation using the Official Algorithm
for Precipitation (OAP) 2.1.1 has been completed. The purpose of this 
recalculation is described in Appendix A of the OAP 2.1 documentation:
https://www1.ncdc.noaa.gov/pub/data/uscrn/documentation/maintenance/2016-05/USCRN_OAP2.1.Description_PrecipChanges.pdf

  ##########################
  
Status change:      Correction
Correction Applied: 2017-05-01
Variables impacted: Precipitation for 3 stations
	
    From July 2016 until May 1, 2017, the precipitation values for 
the following stations and time periods were mistakenly missing on 
the website and in the FTP product files:
		NC Asheville 8 SSW          2004-05-01 to 2004-11-01 
		SC McClellanville 7 NE      2005-05-01 to 2005-11-01
		CA Stovepipe Wells 1 SW     2004-05-01 to 2005-06-01
Precipitation data for these stations/time periods that were downloaded
during the affected time period should be re-acquired. 

  ##########################
  
Version change:     2.0 to 2.1 
Commencement Date:  2016-05-12 
Completion Date:    2016-06-13
Variables impacted: Precipitation
                    Relative Humidity (RH)
                    Temperatures from RH Sensor
                    Thermometer Shield Aspiration Fan Speeds
                    Air Temperature (when fan speeds are low)
                      
    Beginning May 12, 2016, USCRN changed the current data set to 
version 2.1 from v2.0. This version change was retroactively applied to 
all USCRN/USRCRN stations for their period-of-record 5-minute measurements 
from 2016-05-12 until 2016-06-13. [Note that during this period of 
reprocessing, data on the website and in these FTP products contained 
a mixture of v2.0 and v2.1 values.]
    Version 2.1 includes minor precipitation algorithm changes and 
changes/additions to the quality control ranges for acceptable 
relative humidity (RH) values, temperatures measured with the 
RH sensors, and for the speed of the fans which are used to 
aspirate the air temperature sensors. For precipitation, the Official 
Algorithm for Precipitation (OAP) v2.1 was implemented which  
addresses a minor correction to v2.0 that guards against
overally large (> 0.3 mm) precipitation residuals from one hour 
being transferred to the next hour. Further information can be found at 
http://www.ncdc.noaa.gov/crn/documentation.html.

  ##########################
  
Version change:     1.0 to 2.0 
Commencement Date:  2015-08-17
Completion Date:    2015-09-15
Variables impacted: Precipitation

    The original Official Algorithm for Precipitation 
(OAP) version 1.0 was operational until August 17, 2015 and used a 
pairwise comparison and moving reference depth to calculate 
precipitation. Precipitation data accessed and/or downloaded 
prior to this date were calculated using OAP v1.0. 
    Beginning August 17, 2015, all precipitation data were 
calculated using a new processing algorithm, OAP v2.0. In addition, 
the v2.0 algorithm was retroactively applied to all USCRN/USRCRN 
stations for their periods of record (PORs) starting when 5-minute 
data began being collected. The reprocessing took approximately four
weeks to recalculate all station's existing values from v1.0 to v2.0
for their PORs and was completed on September 15, 2015. 
    OAP v2.0 marked a fundamental shift in the procedures used to calculate 
precipitation. The new algorithm uses a weighted average approach 
based on each sensor's noise level. It has greatly improved the 
network's capacity to detect lighter precipitation with greater 
confidence. For details, see Leeper et. al., 2015, 
(http://journals.ametsoc.org/doi/abs/10.1175/JTECH-D-14-00185.1) and
http://www.ncdc.noaa.gov/crn/documentation.html.

********************************************************************************

3. DIRECTORY / FILE ORGANIZATION

The daily01 directory contains the following subdirectories:

    - /2000 through the present year 
        -- useful for obtaining the most up-to-date data from one or more 
        stations for a particular year or years (typically updated on a daily 
        basis).
    
    - /snapshots 
        -- useful for obtaining data from all stations for all years. 
        The entire dataset (as of the date/time of the filename) can be 
        downloaded in a single compressed file. New snapshot files are created 
        on a regular basis; the most recent file contains the most recent set 
        of data.
    
    - /updates 
        -- real-time files broadcast over NOAAPort; stored in yearly 
        subdirectories. These files have the daily values for all stations 
        processed that day.
    
    - /obsolete
        -- files which are now considered to be obsolete and unsuitable for 
        scientific uses but are available as a historical record of work.

********************************************************************************
        
4. FILE AND FILENAME FORMATS

  A. YEARLY

    Each yearly subdirectory in daily01 contains one file for each USCRN/USRCRN 
    station listing its daily data in ASCII text format. The files are named 
    according to the following convention:

                CRND01TT-YYYY-${name}.txt

       CRND01 = filename prefix to denote CRN Daily01 data ** See note below.
           TT = 2-character file format number (currently 03)
         YYYY = 4-digit year
      ${name} = station name (state location vector) (e.g. AZ_Tucson_11_W)

    ** Filename prefix revision:
        On April 11, 2011 15:00 UTC, the prefix 'CRNDAILY' was replaced with
        'CRND01' and changed retroactively for all years.

    The 2-character sequence TT indicates the file format number and is updated 
    when the file format is changed. Previous changes were applied retroactively
    for all years and are outlined below:

     1. [Transition from format '01' to '02']
        On March 22, 2011 15:00 UTC the format of the Daily product was
        changed by removing the COOPNO (Cooperative Observer Program Number)
        column. This had been the second column in the format, starting at 
        character 7 and ending at character 12. With the removal of 
        this column, all following columns were shifted to the left by
        7 characters. The contents of the shifted columns remained the same.

     2. [Transition from format '02' to '03']
        On January 7, 2013 15:00 UTC the format of the Daily product was
        changed by adding the SUR_TEMP_DAILY_TYPE column. This became the 
        12th column in the format, starting and ending at character 88. 
        With the addition of this column, all the following columns were 
        shifted to the right by 2 characters. The contents of the shifted 
        columns remained the same.
                
  B. SNAPSHOTS

    The 'snapshots' directory contains a series of compressed (.zip) files. 
    Each file contains the entire data archive of USCRN/USRCRN daily01 data as 
    of the date/time shown on its filename. Once uncompressed, the directory and
    file structure are identical to the yearly subdirectories (as explained 
    above). Snapshots are created and uploaded at regular intervals, and are 
    named according to the following convention:

                CRND01TT-YYYYMMDDHHmm.zip
    
       CRND01 = filename prefix to denote CRN Daily01 data ** See note below.
           TT = 2-character file format number (see above)
         YYYY = 4-digit year files generated
           MM = 2-digit month files generated (01=Jan, ..., 12=Dec)
           DD = 2-digit day of month files generated
           HH = 2-digit hour of day files generated
           mm = 2-digit minute files generated

    ** Filename prefix revision:
        Snapshot files created prior to April 11, 2011 15:00 UTC, use the 
        'CRNDAILY' prefix.
        
    Users may obtain the most recent archive of data for all stations and all 
    years by downloading the most recent file in this directory. As improvements
    to the dataset are made, the snapshot files serve as a historical record of 
    the archive through time.
    
  C. UPDATES
                
    The 'updates' directory contains a record of the real-time files which are 
    produced daily and broadcast over NOAAPort (SXXX91 CRND01). Each 
    file contains all the USCRN/USRCRN daily data from every station loaded into
    the database since the last transmission. The yearly subdirectories contain 
    a collection of ASCII files named according to the following convention:

                CRND01TT-YYYYMMDDHHmm.txt

       CRND01 = filename prefix to denote CRN Daily01 data ** See note above.
           TT = 2-character file format number (see above)
         YYYY = 4-digit local year
           MM = 2-digit local month (01=Jan, ..., 12=Dec)
           DD = 2-digit local day of month
           HH = 2-digit local standard time hour of day (always 23)
           mm = 2-digit minute (always 59)

    The 'updates' files are named according to the calendar date of the
    observations. The daily records for the previous calendar day are generated 
    several times each day to ensure that late arriving data are included.
    
********************************************************************************

5. DATA FIELDS / FORMATS

Each station file contains fixed-width formatted fields with a single day's 
data per line. A summary table of the fields and a detailed listing of field 
definitions/column formats are shown below. 

Fortran users will find the column widths and counts useful. 

The file "HEADERS.txt", found in the daily01 directory, is designed to be 
prepended to the data for use with spreadsheet programs, data extraction tools 
(e.g. awk) or any other programming language. This file contains the following 
three lines:

    1. Field Number
    2. Field Name
    3. Unit of Measure

Please be sure to refer to the "Important Notes" section below for essential 
information.

All daily data are calculated over the station's 24-hour LST day.

Field#  Name                           Units
---------------------------------------------
   1    WBANNO                         XXXXX
   2    LST_DATE                       YYYYMMDD
   3    CRX_VN                         XXXXXX
   4    LONGITUDE                      Decimal_degrees
   5    LATITUDE                       Decimal_degrees
   6    T_DAILY_MAX                    Celsius
   7    T_DAILY_MIN                    Celsius
   8    T_DAILY_MEAN                   Celsius
   9    T_DAILY_AVG                    Celsius
   10   P_DAILY_CALC                   mm
   11   SOLARAD_DAILY                  MJ/m^2
   12   SUR_TEMP_DAILY_TYPE            X
   13   SUR_TEMP_DAILY_MAX             Celsius
   14   SUR_TEMP_DAILY_MIN             Celsius
   15   SUR_TEMP_DAILY_AVG             Celsius
   16   RH_DAILY_MAX                   %
   17   RH_DAILY_MIN                   %
   18   RH_DAILY_AVG                   %
   19   SOIL_MOISTURE_5_DAILY          m^3/m^3
   20   SOIL_MOISTURE_10_DAILY         m^3/m^3
   21   SOIL_MOISTURE_20_DAILY         m^3/m^3
   22   SOIL_MOISTURE_50_DAILY         m^3/m^3
   23   SOIL_MOISTURE_100_DAILY        m^3/m^3
   24   SOIL_TEMP_5_DAILY              Celsius
   25   SOIL_TEMP_10_DAILY             Celsius
   26   SOIL_TEMP_20_DAILY             Celsius
   27   SOIL_TEMP_50_DAILY             Celsius
   28   SOIL_TEMP_100_DAILY            Celsius

   1    WBANNO  [5 chars]  cols 1 -- 5 
          The station WBAN number.

   2    LST_DATE  [8 chars]  cols 7 -- 14 
          The Local Standard Time (LST) date of the observation.

   3    CRX_VN  [6 chars]  cols 16 -- 21 
          The version number of the station datalogger program that was in 
          effect at the time of the observation. Note: This field should be 
          treated as text (i.e. string).

   4    LONGITUDE  [7 chars]  cols 23 -- 29 
          Station longitude, using WGS-84.

   5    LATITUDE  [7 chars]  cols 31 -- 37 
          Station latitude, using WGS-84.

   6    T_DAILY_MAX  [7 chars]  cols 39 -- 45 
          Maximum air temperature, in degrees C. See Note F.

   7    T_DAILY_MIN  [7 chars]  cols 47 -- 53 
          Minimum air temperature, in degrees C. See Note F.

   8    T_DAILY_MEAN  [7 chars]  cols 55 -- 61 
          Mean air temperature, in degrees C, calculated using the typical 
          historical approach: (T_DAILY_MAX + T_DAILY_MIN) / 2. See Note F.

   9    T_DAILY_AVG  [7 chars]  cols 63 -- 69 
          Average air temperature, in degrees C. See Note F.

   10   P_DAILY_CALC  [7 chars]  cols 71 -- 77 
          Total amount of precipitation, in mm. See Note F.

   11   SOLARAD_DAILY  [8 chars]  cols 79 -- 86 
          Total solar energy, in MJ/meter^2, calculated from the hourly average 
          global solar radiation rates and converted to energy by integrating 
          over time.

   12   SUR_TEMP_DAILY_TYPE  [1 chars]  cols 88 -- 88 
          Type of infrared surface temperature measurement. 'R' denotes raw 
          measurements, 'C' denotes corrected measurements, and 'U' indicates 
          unknown/missing. See Note G.

   13   SUR_TEMP_DAILY_MAX  [7 chars]  cols 90 -- 96 
          Maximum infrared surface temperature, in degrees C.

   14   SUR_TEMP_DAILY_MIN  [7 chars]  cols 98 -- 104 
          Minimum infrared surface temperature, in degrees C.

   15   SUR_TEMP_DAILY_AVG  [7 chars]  cols 106 -- 112 
          Average infrared surface temperature, in degrees C.

   16   RH_DAILY_MAX  [7 chars]  cols 114 -- 120 
          Maximum relative humidity, in %. See Notes H and I.

   17   RH_DAILY_MIN  [7 chars]  cols 122 -- 128 
          Minimum relative humidity, in %. See Notes H and I.

   18   RH_DAILY_AVG  [7 chars]  cols 130 -- 136 
          Average relative humidity, in %. See Notes H and I.

   19   SOIL_MOISTURE_5_DAILY  [7 chars]  cols 138 -- 144 
          Average soil moisture, in fractional volumetric water content (m^3/m^3), 
          at 5 cm below the surface. See Notes I and J.

   20   SOIL_MOISTURE_10_DAILY  [7 chars]  cols 146 -- 152 
          Average soil moisture, in fractional volumetric water content (m^3/m^3), 
          at 10 cm below the surface. See Notes I and J.

   21   SOIL_MOISTURE_20_DAILY  [7 chars]  cols 154 -- 160 
          Average soil moisture, in fractional volumetric water content (m^3/m^3), 
          at 20 cm below the surface. See Notes I and J.

   22   SOIL_MOISTURE_50_DAILY  [7 chars]  cols 162 -- 168 
          Average soil moisture, in fractional volumetric water content (m^3/m^3), 
          at 50 cm below the surface. See Notes I and J.

   23   SOIL_MOISTURE_100_DAILY  [7 chars]  cols 170 -- 176 
          Average soil moisture, in fractional volumetric water content (m^3/m^3), 
          at 100 cm below the surface. See Notes I and J.

   24   SOIL_TEMP_5_DAILY  [7 chars]  cols 178 -- 184 
          Average soil temperature, in degrees C, at 5 cm below the surface. 
          See Notes I and J.

   25   SOIL_TEMP_10_DAILY  [7 chars]  cols 186 -- 192 
          Average soil temperature, in degrees C, at 10 cm below the surface. 
          See Notes I and J.

   26   SOIL_TEMP_20_DAILY  [7 chars]  cols 194 -- 200 
          Average soil temperature, in degrees C, at 20 cm below the surface. 
          See Notes I and J.

   27   SOIL_TEMP_50_DAILY  [7 chars]  cols 202 -- 208 
          Average soil temperature, in degrees C, at 50 cm below the surface. 
          See Notes I and J.

   28   SOIL_TEMP_100_DAILY  [7 chars]  cols 210 -- 216 
          Average soil temperature, in degrees C, at 100 cm below the surface. 
          See Notes I and J.

    IMPORTANT NOTES:
        A.  All fields are separated from adjacent fields by at least one space.
        B.  Leading zeros are omitted.
        C.  Missing data are indicated by the lowest possible integer for a 
            given column format, such as -9999.0 for 7-character fields with 
            one decimal place or -99.000 for 7-character fields with three
            decimal places.
        D.  Daily data are calculated using the station's local day. 
        E.  There are no quality flags for these derived quantities. When the 
            raw data are flagged as erroneous, these derived values are not 
            calculated, and are instead reported as missing. Therefore, these 
            fields may be assumed to always be good (unflagged) data, except 
            when they are reported as missing.
        F.  The daily values reported in this dataset are calculated using 
            multiple independent measurements for temperature and precipitation. 
            USCRN/USRCRN stations have multiple co-located temperature sensors 
            that make 10-second independent measurements which are used to 
            produce max/min/avg temperature values at 5-minute intervals. The
            precipitation gauge is equipped with multiple load cell sensors to 
            provide independent measurements of depth change at 5-minute 
            intervals. 
        G.  On 2013-01-07 at 1500 UTC, USCRN began reporting corrected surface 
            temperature measurements for some stations. These changes  
            impact previous users of the data because the corrected values 
            differ from uncorrected values. To distinguish between uncorrected 
            (raw) and corrected surface temperature measurements, a surface 
            temperature type field was added to the daily01 product. The 
            possible values of the this field are "R" to denote raw surface 
            temperature measurements, "C" to denote corrected surface 
            temperature measurements, and "U" for unknown/missing.
        H.  Relative humidity is computed from 5-minute values in almost all 
            cases. All USCRN stations now report 5-minute averages, however the 
            two Asheville, NC stations reported only hourly RH values until 
            2007-02-22.
        I.  USRCRN stations do not measure solar radiation, surface temperature,
            relative humidity or soil variables, so those fields are shown as 
            missing data.
        J.  USCRN stations have multiple co-located soil sensors that record 
            independent measurements. The soil values reported in this dataset 
            are an average of the day's hourly soil measurements which are 
            calculated from the multiple independent measurements. Soil 
            moisture is the ratio of water volume over sample volume 
            (m^3 water/m^3 soil).
        K.  In accordance with Service Change Notice 14-25 from the National 
            Weather Service, NCDC stopped providing data from the 72 
            Southwest Regional Climate Reference Network (USRCRN) stations on 
            June 1, 2014. The historical data for these stations remain 
            available. 