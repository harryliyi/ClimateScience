;*****************************************************
;*****************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "./taylor_diagram.ncl"
load "./taylor_stats.ncl"
;*****************************************************
begin
;*****************************************************
; open files and read in monthly data
;*****************************************************
  dir1   = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/vrseasia_AMIP_1979_to_2005/atm/hist/"
  dir2   = "/scratch/d/dylan/harryli/gpcdata/cesm1.2/gpc_cesm1_2_2/archive/AMIP1979control_2deg/atm/hist/"
  dir3   = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/f09_f09_AMIP_1979_to_2005/atm/hist/"
  dir4   = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/f19_f19_AMIP_1979_to_2005/atm/hist/"
;  dir4   = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/ne30_ne30_AMIP_1979_to_2005/atm/hist/"
  obdir1   = "/scratch/d/dylan/harryli/obsdataset/CRU/"
  obdir2   = "/scratch/d/dylan/harryli/obsdataset/GPCP/"
  obdir3   = "/scratch/d/dylan/harryli/obsdataset/GPCC/"
  rdir   = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/dst/"
  odir   = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_obs/"

  f1     = addfile (dir1+"fv02_prec_vrseasia_AMIP_1979_to_2005.cam.h0.1979-2005.nc", "r")
  f2     = addfile (dir2+"diag_AMIP1979control_2deg.cam.h0.1979_2005.nc", "r")
  f3     = addfile (dir3+"PREC_f09_f09_AMIP_1979_to_2005.cam.h0.1979-2005.nc","r")
  f4     = addfile (dir4+"PREC_f19_f19_AMIP_1979_to_2005.cam.h0.1979-2005.nc","r")
;  f4     = addfile (dir4+"fv09_PREC_ne30_ne30_AMIP_1979_to_2005.cam.h0.1979-2005.nc","r")
  obf1   = addfile (obdir1+"cru_ts3.21.1901.2012.pre_orig_mmday.nc", "r")
  obf2   = addfile (obdir2+"gpcp_cdr_v23rB1_197901-201608.nc", "r")
  obf3   = addfile (obdir3+"precip.mon.total.v7.nc", "r")
  rf     = addfile (rdir+"dst_1.9x2.5_c090203.nc","r")
  
  iniyear = 1980
  endyear = 2005

  print("Start loading data...")

  vrm1  = f1->PRECT((iniyear-1979)*12:(endyear-1979)*12+11,:,:)
  vrm2  = f2->PRECT((iniyear-1979)*12:(endyear-1979)*12+11,:,:)
  vrm3  = f3->PRECT((iniyear-1979)*12:(endyear-1979)*12+11,:,:)
  vrm4  = f4->PRECT((iniyear-1979)*12:(endyear-1979)*12+11,:,:)
  vrm5  = obf1->pre((iniyear-1901)*12:(endyear-1901)*12+11,:,:)
  vrm6  = obf2->precip((iniyear-1979)*12:(endyear-1979)*12+11,:,:)
;  vrm7  = obf3->precip(1128:1211,:,:)
;  vrm7 = vrm7(:,::-1,:)

  print("Loading data:Done!")

  lon1    = f1->lon
  lat1    = f1->lat
  lon2    = f2->lon
  lat2    = f2->lat
  lon3    = f3->lon
  lat3    = f3->lat
  lon4    = f4->lon
  lat4    = f4->lat
  lon5    = obf1->lon
  lat5    = obf1->lat
  lon6    = obf2->longitude
  lat6    = obf2->latitude
;  lon7    = obf3->lon
;  lat7    = obf3->lat
  rlon    = rf->lon
  rlat    = rf->lat

;  lat7    = lat7(::-1)

;  print(lon5)
;  print(lat5)

  yeart = ispan(iniyear,endyear,1)
  montht = ispan(1,(endyear-iniyear+1)*12,1)
  m=(/01,02,03,04,05,06,07,08,09,10,11,12/)
  mdays =(/31,28,31,30,31,30,31,31,30,31,30,31/)
  month = (/"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"/)
;*****************************************************
; some parameters
;***************************************************** 
  nyrs   = endyear-iniyear+1
  dsizes = dimsizes(vrm2)
  ntime  = dsizes(0)
  nlat   = dimsizes(rlat)
  nlon   = dimsizes(rlon)
;*****************************************************   
; Regriding
;*****************************************************
  print("Start regriding data...")
;  vrm1  = lonFlip(vrm1)
;  vrm2  = lonFlip(vrm2)

  vrm1!0 = "time"
  vrm1!1 = "lat"
  vrm1!2 = "lon"
  vrm1&lat = lat1
  vrm1&lon = lon1

  vrm2!0 = "time"
  vrm2!1 = "lat"
  vrm2!2 = "lon"
  vrm2&lat = lat2
  vrm2&lon = lon2

  vrm3!0 = "time"
  vrm3!1 = "lat"
  vrm3!2 = "lon"
  vrm3&lat = lat3
  vrm3&lon = lon3

  vrm4!0 = "time"
  vrm4!1 = "lat"
  vrm4!2 = "lon"
  vrm4&lat = lat4
  vrm4&lon = lon4

  vrm5!0 = "time"
  vrm5!1 = "lat"
  vrm5!2 = "lon"
  vrm5&lat = lat5
  vrm5&lon = lon5

  vrm6!0 = "time"
  vrm6!1 = "lat"
  vrm6!2 = "lon"
  vrm6&lat = lat6
  vrm6&lon = lon6

;do i=1,120
;  k=mod(i,12)
;  if (k.eq.0) then
;     k=12
;  end if
;  vrm7(i-1,:,:)=vrm7(i-1,:,:)/mdays(k-1)
;end do

;  vrm7!0 = "time"
;  vrm7!1 = "lat"
;  vrm7!2 = "lon"
;  vrm7&lat = lat7
;  vrm7&lon = lon7

;  vrm3 = lonFlip(vrm3)
;  lon3 = lonFlip(lon3)

  vrmt = vrm5
  vrmt(:,:,0:359) = vrm5(:,:,360:719)
  vrmt(:,:,360:719) = vrm5(:,:,0:359)
  lont = lon5
  lont(0:359) = lon5(360:719)
  lont(360:719) = lon5(0:359) + 360
  
  opt = False
;  print(lat1)
;  print(lat2)
;  print(lat3)
;  print(lon4)
;  vr5   = area_conserve_remap_Wrap(lon5,lat2,vrm2,lon1,lat1,opt)
;  vr1 = linint2 (lon1,lat1,vrm1,True,rlon,rlat,0)
;  vr2 = linint2 (lon2,lat2,vrm2,True,rlon,rlat,0)
  vr1 = linint2 (lon1,lat1,vrm1,True,rlon,rlat,0)
;  vr2 = linint2 (lon2,lat2,vrm2,True,rlon,rlat,0)
  vr3 = linint2 (lon3,lat3,vrm3,True,rlon,rlat,0)
  vr4 = linint2 (lon4,lat4,vrm4,True,rlon,rlat,0)
  vr5 = linint2 (lont,lat5,vrmt,True,rlon,rlat,0)
  vr6 = linint2 (lon6,lat6,vrm6,True,rlon,rlat,0)
;  print(vr2)

;  vr1   = vrm1
  vr2   = vrm2
;  vr3   = vrm3
;  vr4   = vrm4

  vr1 = mask(vr1,vr5.lt.1000,True)
  vr2 = mask(vr2,vr5.lt.1000,True)
  vr3 = mask(vr3,vr5.lt.1000,True)
  vr4 = mask(vr4,vr5.lt.1000,True)
  vr6 = mask(vr6,vr5.lt.1000,True)

  vr1!0 = "time"
  vr1!1 = "lat"
  vr1!2 = "lon"
  vr1&lat = rlat
  vr1&lon = rlon

  vr2!0 = "time"
  vr2!1 = "lat"
  vr2!2 = "lon"
  vr2&lat = rlat
  vr2&lon = rlon

  vr3!0 = "time"
  vr3!1 = "lat"
  vr3!2 = "lon"
  vr3&lat = rlat
  vr3&lon = rlon
 
  vr4!0 = "time"
  vr4!1 = "lat"
  vr4!2 = "lon"
  vr4&lat = rlat
  vr4&lon = rlon

  vr5!0 = "time"
  vr5!1 = "lat"
  vr5!2 = "lon"
  vr5&lat = rlat
  vr5&lon = rlon

  vr6!0 = "time"
  vr6!1 = "lat"
  vr6!2 = "lon"
  vr6&lat = rlat
  vr6&lon = rlon

;  vr1  = lonFlip(vr1)
;  vr2  = lonFlip(vr2)
;  vr3  = lonFlip(vr3)

;  vrmd  = vr1-vr2
;  vrmd  = 86400*1000*vrmd

  vr1 = 86400*1000*vr1
  vr2 = 86400*1000*vr2
  vr3 = 86400*1000*vr3
  vr4 = 86400*1000*vr4

  print("Regriding data:Done!")

;*****************************************************
; Calculate taylor stats
;*****************************************************
  case      = (/ "Mainland SEA", "North Amercia" /)
  var       = (/ "vrseasia","fv1.9x2.5","fv0.9x1.25","ne30","GPCP" /)   

;  vrsea1 = dim_avg_n_Wrap(vr1(:,{10:25},{100:110}),0)
;  vrsea2 = dim_avg_n_Wrap(vr2(:,{10:25},{100:110}),0)
;  vrsea3 = dim_avg_n_Wrap(vr3(:,{10:25},{100:110}),0)
;  vrsea4 = dim_avg_n_Wrap(vr4(:,{10:25},{100:110}),0)
;  vrsea5 = dim_avg_n_Wrap(vr5(:,{10:25},{100:110}),0)
;  vrsea6 = dim_avg_n_Wrap(vr6(:,{10:25},{100:110}),0)

  vrsea1 = dim_avg_n_Wrap(dim_avg_n_Wrap(vr1(:,{10:25},{100:110}),1),1)
  vrsea2 = dim_avg_n_Wrap(dim_avg_n_Wrap(vr2(:,{10:25},{100:110}),1),1)
  vrsea3 = dim_avg_n_Wrap(dim_avg_n_Wrap(vr3(:,{10:25},{100:110}),1),1)
  vrsea4 = dim_avg_n_Wrap(dim_avg_n_Wrap(vr4(:,{10:25},{100:110}),1),1)
  vrsea5 = dim_avg_n_Wrap(dim_avg_n_Wrap(vr6(:,{10:25},{100:110}),1),1)
  refsea = doubletofloat(dim_avg_n_Wrap(dim_avg_n_Wrap(vr5(:,{10:25},{100:110}),1),1))

;  vrsea1 = mask(vrsea1,refsea.lt.1000,True)
;  vrsea2 = mask(vrsea2,refsea.lt.1000,True)
;  vrsea3 = mask(vrsea3,refsea.lt.1000,True)
;  vrsea4 = mask(vrsea4,refsea.lt.1000,True)
;  vrsea5 = mask(vrsea5,refsea.lt.1000,True)

;  vrsea1@_FillValue = -9999
;  vrsea2@_FillValue = -9999
;  vrsea3@_FillValue = -9999
;  vrsea4@_FillValue = -9999
;  vrsea5@_FillValue = -9999
;  refsea@_FillValue = -9999
  
;  vrsea5 = linmsg(vrsea5,-1)
  
  print(vrsea1)
  print(refsea)

  sea_rat = new (5,double)
  sea_cc  = new (5,double)
 
  sea_cc(0) = escorc(vrsea1,refsea)
  sea_cc(1) = escorc(vrsea2,refsea)
  sea_cc(2) = escorc(vrsea3,refsea)
  sea_cc(3) = escorc(vrsea4,refsea)
  sea_cc(4) = escorc(vrsea5,refsea)
  
  sea_rat(0) = stddev(vrsea1)/stddev(refsea)
  sea_rat(1) = stddev(vrsea2)/stddev(refsea)
  sea_rat(2) = stddev(vrsea3)/stddev(refsea)
  sea_rat(3) = stddev(vrsea4)/stddev(refsea)
  sea_rat(4) = stddev(vrsea5)/stddev(refsea)
 
;***********************************************************************************************************
;  dim = dimsizes(vrsea1)
;  ncol = dim(0)*dim(1)

;  print(vrsea1)
;  print(reshape(vrsea1,ncol))
;  print(pattern_cor(vrsea1,vrsea5,1.0,0))
;  w = 1.0
;  print(pattern_cor(vrsea1(ind(.not.ismissing(vrsea1))), vrsea5(ind(.not.ismissing(vrsea5))), w, 0)) 
 
;  tay_stat = taylor_stats(vrsea1, vrsea5, w, 0) 
;  print(tay_stat)
;  sea_cc(0)  = tay_stat(0)
;  sea_rat(0) = tay_stat(1)
  
;  print(dim_rmsd(reshape(vrsea1,ncol),reshape(vrsea5,ncol)))
;  print(sqrt((dim_rmsd(reshape(vrsea1,ncol),reshape(vrsea5,ncol)))^2+(avg(vrsea1)-avg(vrsea1))^2)/stddev(vrsea5))

;  tay_stat = taylor_stats(vrsea2, vrsea5, w, 0)
;  sea_cc(1)  = tay_stat(0)
;  sea_rat(1) = tay_stat(1)

;  tay_stat = taylor_stats(vrsea3, vrsea5, w, 0)
;  sea_cc(2)  = tay_stat(0)
;  sea_rat(2) = tay_stat(1)

;  tay_stat = taylor_stats(vrsea4, vrsea5, w, 0)
;  sea_cc(3)  = tay_stat(0)
;  sea_rat(3) = tay_stat(1)

;  tay_stat = taylor_stats(vrsea6, vrsea5, w, 0)
;  sea_cc(4)  = tay_stat(0)
;  sea_rat(4) = tay_stat(1)

;  vrna1 = dim_avg_n_Wrap(vr1(:,{30:60},{240:280}),0)
;  vrna2 = dim_avg_n_Wrap(vr2(:,{30:60},{240:280}),0)
;  vrna3 = dim_avg_n_Wrap(vr3(:,{30:60},{240:280}),0)
;  vrna4 = dim_avg_n_Wrap(vr4(:,{30:60},{240:280}),0)
;  vrna5 = dim_avg_n_Wrap(vr5(:,{30:60},{240:280}),0)
;  vrna6 = dim_avg_n_Wrap(vr6(:,{30:60},{240:280}),0)
;***********************************************************************************************************

  vrna1 = dim_avg_n_Wrap(dim_avg_n_Wrap(vr1(:,{30:60},{240:280}),1),1)
  vrna2 = dim_avg_n_Wrap(dim_avg_n_Wrap(vr2(:,{30:60},{240:280}),1),1)
  vrna3 = dim_avg_n_Wrap(dim_avg_n_Wrap(vr3(:,{30:60},{240:280}),1),1)
  vrna4 = dim_avg_n_Wrap(dim_avg_n_Wrap(vr4(:,{30:60},{240:280}),1),1)
  vrna5 = dim_avg_n_Wrap(dim_avg_n_Wrap(vr6(:,{30:60},{240:280}),1),1)
  refna = doubletofloat(dim_avg_n_Wrap(dim_avg_n_Wrap(vr5(:,{30:60},{240:280}),1),1))

;  vrna1 = mask(vrna1,vrna5.lt.1000,True)
;  vrna2 = mask(vrna2,vrna5.lt.1000,True)
;  vrna3 = mask(vrna3,vrna5.lt.1000,True)
;  vrna4 = mask(vrna4,vrna5.lt.1000,True)
;  vrna5 = mask(vrna5,vrna5.lt.1000,True)

;  vrna1@_FillValue = -9999
;  vrna2@_FillValue = -9999
;  vrna3@_FillValue = -9999
;  vrna4@_FillValue = -9999
;  vrna5@_FillValue = -9999
;  refna@_FillValue = -9999

  na_rat = new (5,double)
  na_cc  = new (5,double)

  na_cc(0) = escorc(vrna1,refna)
  na_cc(1) = escorc(vrna2,refna)
  na_cc(2) = escorc(vrna3,refna)
  na_cc(3) = escorc(vrna4,refna)
  na_cc(4) = escorc(vrna5,refna)

  na_rat(0) = stddev(vrna1)/stddev(refna)
  na_rat(1) = stddev(vrna2)/stddev(refna)
  na_rat(2) = stddev(vrna3)/stddev(refna)
  na_rat(3) = stddev(vrna4)/stddev(refna)
  na_rat(4) = stddev(vrna5)/stddev(refna)

;  dim = dimsizes(vrna1)
;  ncol = dim(0)*dim(1)

;  tay_stat = taylor_stats(vrna1, vrna5, w, 0)
;  na_cc(0)  = tay_stat(0)
;  na_rat(0) = tay_stat(1)

;  tay_stat = taylor_stats(vrna2, vrna5, w, 0)
;  na_cc(1)  = tay_stat(0)
;  na_rat(1) = tay_stat(1)

;  tay_stat = taylor_stats(vrna3, vrna5, w, 0)
;  na_cc(2)  = tay_stat(0)
;  na_rat(2) = tay_stat(1)

;  tay_stat = taylor_stats(vrna4, vrna5, w, 0)
;  na_cc(3)  = tay_stat(0)
;  na_rat(3) = tay_stat(1)

;  tay_stat = taylor_stats(vrna6, vrna5, w, 0)
;  na_cc(4)  = tay_stat(0)
;  na_rat(4) = tay_stat(1)

  nDataSets  = 2                               ; number of datasets
  npts       = dimsizes(sea_rat)
  ratio      = new ((/nDataSets, npts/),typeof(sea_rat) )
  cc         = new ((/nDataSets, npts/),typeof(sea_cc ) )

  ratio(0,:) = sea_rat
  cc(0,:)    = sea_cc 

  ratio(1,:) = na_rat
  cc(1,:)    = na_cc 

  print(ratio)
  print(cc)
;**********************************
; create taylor plot
;**********************************
  res                 = True                     ; diagram mods desired
  res@tiMainString    = "prect taylor diagram (ref as cru)"                ; title
  res@tiMainFontHeightF   = 0.02
  res@tmXBLabelFontHeightF = 0.015                 ; resize tick labels
  res@tmYLLabelFontHeightF = 0.015
  
  res@Colors          = (/"red","blue"/)         ; marker colors
  res@Markers         = (/14,9/)                 ; marker styles
  res@markerTxYOffset = 0.04                     ; offset btwn marker & label
  res@gsMarkerSizeF   = 0.01                     ; marker size   
  res@txFontHeightF   = 0.015                    ; text size 

  res@stnRad          = (/ 0.5, 1.5 /)     ; additional standard radii
  res@ccRays          = (/ 0.6, 0.9 /)     ; correlation rays
  res@ccRays_color    = "LightGray"        ; default is "black"

  res@centerDiffRMS   = True               ; RMS 'circles'
  res@centerDiffRMS_color = "LightGray"    ; default is "black"

  res@varLabels    = var
  res@caseLabels   = case

  wks   = gsn_open_wks("pdf",odir+"vrseasia_prect_taylor_ts_ref_CRU")     ; send graphics to PNG file 
  plot  = taylor_diagram(wks, ratio, cc, res)
  
end
