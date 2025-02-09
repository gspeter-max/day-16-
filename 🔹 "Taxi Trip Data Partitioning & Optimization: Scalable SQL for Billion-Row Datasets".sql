🔹 "Taxi Trip Data Partitioning & Optimization: Scalable SQL for Billion-Row Datasets"

create table texi_trip (
    id bigint auto_increment primary key , 
    pickup_datetime datetime not null , 
    pickup_location_id int not null, 
    index  idx_pickup_datetime (pickup_datetime), 
    index idx_pickup_location_id (pickup_location_id)
)engine = InnoDB; 


delimiter $$
create procedure insertTexiData()
    begin 
        declare i int default 0 ; 
        while i < 1000000 do 
            insert into texi_trip (pickup_datetime , pickup_location_id)
            select now() - interval floor(rand() * 3650) day , floor(rand() * 500); 
            set i = i + 1 ; 
        end while; 
end $$
delimiter ; 

call insertTexiData(); 

create table texi_trip_partition (
    id bigint auto_increment primary key , 
    pickup_datetime datetime not null, 
    pickup_location_id int  not null, 
    pickup_hour int generated always as (hour(pickup_datetime)) STORED, 
    index idx_index ( pickup_datetime, pickup_location_id )
    ) partition by range ( pickup_hour ) (
        
    PARTITION p00 VALUES LESS THAN (1),
    PARTITION p01 VALUES LESS THAN (2),
    PARTITION p02 VALUES LESS THAN (3),
    PARTITION p03 VALUES LESS THAN (4),
    PARTITION p04 VALUES LESS THAN (5),
    PARTITION p05 VALUES LESS THAN (6),
    PARTITION p06 VALUES LESS THAN (7),
    PARTITION p07 VALUES LESS THAN (8),
    PARTITION p08 VALUES LESS THAN (9),
    PARTITION p09 VALUES LESS THAN (10),
    PARTITION p10 VALUES LESS THAN (11),
    PARTITION p11 VALUES LESS THAN (12),
    PARTITION p12 VALUES LESS THAN (13),
    PARTITION p13 VALUES LESS THAN (14),
    PARTITION p14 VALUES LESS THAN (15),
    PARTITION p15 VALUES LESS THAN (16),
    PARTITION p16 VALUES LESS THAN (17),
    PARTITION p17 VALUES LESS THAN (18),
    PARTITION p18 VALUES LESS THAN (19),
    PARTITION p19 VALUES LESS THAN (20),
    PARTITION p20 VALUES LESS THAN (21),
    PARTITION p21 VALUES LESS THAN (22),
    PARTITION p22 VALUES LESS THAN (23),
    PARTITION p23 VALUES LESS THAN (24)


    );

create table top_pickup_location as
    select date_format(pickup_datetime , '%Y-%m-%d %H:00:00') as hour, 
    pickup_location_id, 
    count(*) as total_pic_count 
from texi_trip_partition
group by hour, pickup_location_id;


delimiter $$

create event refresh_top_pickups 
    on schedule every 1 hour 
    do 
        replace into top_pickup_location 
        select 
            date_format (pickup_datetime, '%Y-%m-%d %H:00:00') as hour, 
            pickup_location_id, 
            count(*) as total_pic_count
        from texi_trip_partition
        group by hour, pickup_location_id
        on duplicate key update total_pic_count = VALUES(total_pic_count);   
$$

delimiter ;

select hour, 
    pickup_location_id, 
    total_pic_count
from top_pickup_location
order by hour, total_pic_count desc 
limit 5 ; 
