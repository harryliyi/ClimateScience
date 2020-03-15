;*****************************************************
;*****************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
;*****************************************************
begin
;*****************************************************
; open files and read in monthly data
;*****************************************************
  dir1   = "/scratch/d/djones/harryli/cesm1.2/VRcesm/cesm1_2_2/archive/AMIP_VRcesm_diag_1992_2005/atm/hist/"
  dir2   = "/scratch/d/djones/harryli/cesm1.2/gpc_cesm1_2_2/archive/AMIPdiag_1920_today_cesm122_defaultini/atm/hist/"
;  dir3   = "/scratch/d/djones/harryli/obsdataset/cru/"
;  dir4   = "/scratch/d/djones/harryli/obsdataset/GPCP/"
  dir5 = "/scratch/d/djones/harryli/obsdataset/GPCC/"
  rdir   = "/scratch/d/djones/harryli/cesm1.2/inputdata/atm/cam/dst/"
  f1     = addfile (dir1+"fv05_AMIP_VRcesm_diag_1992_2005.cam.h0.1992-2004.nc", "r")
  f2     = addfile (dir2+"prect_AMIPdiag_1920_today_cesm122_defaultini.cam.h0.199001-200611.nc", "r")
;  f3     = addfile (dir3+"cru_ts3.21.1901.2012.pre_orig_mmday.nc","r")
;  f4     = addfile (dir4+"gpcp_cdr_v23rB1_197901-201608.nc","r")
  f5     = addfile (dir5+"precip.mon.total.v7.nc","r")
  rf     = addfile (rdir+"dst_1.9x2.5_c090203.nc","r")

  print("Start loading data...")
  vrm1  = f1->PRECT(36:155,:,:)
  vrm2  = f2->PRECT(60:179,:,:)
;  vrm3  = f3->pre(1128:1247,:,:)
;  vrm4  = f4->precip(192:311,:,:)
  vrm5 = f5->precip(1128:1247,:,:)
  vrm5 = vrm5(:,::-1,:)
  print("Loading data:Done!")

  lon1    = f1->lon
  lat1    = f1->lat
  lon2    = f2->lon
  lat2    = f2->lat
;  lon3    = f3->lon
;  lat3    = f3->lat
;  lon4    = f4->longitude
;  lat4    = f4->latitude
  lon5    = f5->lon
  lat5    = f5->lat
  lat5    = lat5(::-1)
  rlon    = rf->lon
  rlat    = rf->lat
  print(lon5)
  print(lat5)

  iniyear = 1995
  endyear = 2004
  yeart = ispan(1995,2004,1)
  montht = ispan(1,120,1)
  m=(/01,02,03,04,05,06,07,08,09,10,11,12/)
  mdays =(/31,28,31,30,31,30,31,31,30,31,30,31/)
  month = (/"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"/)
;*****************************************************
; some parameters
;***************************************************** 
  nyrs   = 5
  dsizes = dimsizes(vrm2)
  ntime  = dsizes(0)
  nlat   = dimsizes(rlat)
  nlon   = dimsizes(rlon)
;*****************************************************   
; Calculation
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

;  vrm3!0 = "time"
;  vrm3!1 = "lat"
;  vrm3!2 = "lon"
;  vrm3&lat = lat3
;  vrm3&lon = lon3

;  vrm4!0 = "time"
;  vrm4!1 = "lat"
;  vrm4!2 = "lon"
;  vrm4&lat = lat4
;  vrm4&lon = lon4

do i=1,120
  k=mod(i,12)
  if (k.eq.0) then
     k=12
  end if
  vrm5(i-1,:,:)=vrm5(i-1,:,:)/mdays(k-1)
end do

  vrm5!0 = "time"
  vrm5!1 = "lat"
  vrm5!2 = "lon"
  vrm5&lat = lat5
  vrm5&lon = lon5

;  vrm3 = lonFlip(vrm3)
;  lon3 = lonFlip(lon3)

;  vrm5 = vrm4
;  vrm5(:,:,0:71) = vrm4(:,:,72:143)
;  vrm5(:,:,72:143) = vrm4(:,:,0:71)
;  lon5 = lon4
;  lon5(0:71) = lon4(72:143)
;  lon5(72:143) = lon4(0:71) + 360
  
  opt = False
;  print(lat1)
;  print(lat2)
;  print(lat3)
;  print(lon4)
;  vr2   = area_conserve_remap_Wrap(lon2,lat2,vrm2,lon1,lat1,opt)
  vr1 = linint2 (lon1,lat1,vrm1,True,rlon,rlat,0)
  vr2 = linint2 (lon2,lat2,vrm2,True,rlon,rlat,0)
  vr3 = linint2 (lon5,lat5,vrm5,True,rlon,rlat,0)
;  print(vr2)
;  vr2   = vrm2

  vr1!0 = "time"
  vr1!1 = "lat"
  vr1!2 = "lon"
  vr1&lat = rlat
  vr1&lon = rlon

  vr2!0 = "time"
  vr2!1 = "lat"
  vr2!2 = "lon"
  vr2&lat = lat2
  vr2&lon = lon2

  vr3!0 = "time"
  vr3!1 = "lat"
  vr3!2 = "lon"
  vr3&lat = rlat
  vr3&lon = rlon
 
;  vr1  = lonFlip(vr1)
;  vr2  = lonFlip(vr2)
;  vr3  = lonFlip(vr3)

;  vrmd  = vr1-vr2
;  vrmd  = 86400*1000*vrmd

  vr1 = 86400*1000*vr1
  vr2 = 86400*1000*vr2

  print("Regriding data:Done!")

do i =0,11
  vryrgseason = new((/nlat,nlon/),double)
  vryrgseason(:,:) = abs(dim_avg_n_Wrap(vr1(i:120+i-12:12,:,:),0)-dim_avg_n_Wrap(vr3(i:120+i-12:12,:,:),0))-abs(dim_avg_n_Wrap(vr2(i:120+i-12:12,:,:),0)-dim_avg_n_Wrap(vr3(i:120+i-12:12,:,:),0))
 
  vryrgseason!0 = "lat"
  vryrgseason!1 = "lon"
  vryrgseason&lat = rlat
  vryrgseason&lon = rlon 
  vryrgseason@units = "mm/day"
;***************************************************** 
; create plots
;***************************************************** 
  ofilename = "AMIP_VRcesm_diag_1992_2005_PRECT_CONTOUR_vsCESM1_2_2_vsGPCC_1995-2004"
  titlename = "Total precipitation bias (VRseasia-cesm1_2_2) 1995-2004 "+month(i)
  wks   = gsn_open_wks ("png", ofilename+"_"+m(i))       ; open ps file
  gsn_define_colormap(wks,"gui_default")        ; choose color map

  res                 = True                    ; plot mods desired
  res@tiMainString    = titlename        ; title
  res@cnFillOn        = True                    ; turn on color
  res@cnFillPalette   = "BlWhRe"              ; choose colormap
  res@cnLinesOn       = False
  res@cnLevelSelectionMode = "ManualLevels"        ; set manual contour levels
  res@cnMinLevelValF       =   -5.25                 ; set min contour level
  res@cnMaxLevelValF       =   5.25                  ; set max contour level
  res@cnLevelSpacingF      =   0.5                  ; set contour spacing

  plot = gsn_csm_contour_map(wks, vryrgseason(:,:), res) 
  
  delete(vryrgseason)
  print("Current month is: "+month(i))
end do

do i =iniyear,endyear
  vryrgseason = new((/nlat,nlon/),double)
  vryrgseason(:,:) = abs(dim_avg_n_Wrap(vr1((i-iniyear)*12:(i-iniyear)*12+11,:,:),0)-dim_avg_n_Wrap(vr3((i-iniyear)*12:(i-iniyear)*12+11,:,:),0))-abs(dim_avg_n_Wrap(vr2((i-iniyear)*12:(i-iniyear)*12+11,:,:),0)-dim_avg_n_Wrap(vr3((i-iniyear)*12:(i-iniyear)*12+11,:,:),0))

  vryrgseason!0 = "lat"
  vryrgseason!1 = "lon"
  vryrgseason&lat = rlat
  vryrgseason&lon = rlon
  vryrgseason@units = "mm/day"
;*****************************************************
; create plots
;*****************************************************
  ofilename = "AMIP_VRcesm_diag_1992_2005_PRECT_CONTOUR_vsCESM1_2_2_vsGPCC_1995-2004"
  titlename = "Total precipitation bias (VRseasia-cesm1_2_2) Avg "+i
  wks   = gsn_open_wks ("png", ofilename+"_"+i)       ; open ps file
  gsn_define_colormap(wks,"gui_default")        ; choose color map

  res                 = True                    ; plot mods desired
  res@tiMainString    = titlename        ; title
  res@cnFillOn        = True                    ; turn on color
  res@cnFillPalette   = "BlWhRe"              ; choose colormap
  res@cnLinesOn       = False
  res@cnLevelSelectionMode = "ManualLevels"        ; set manual contour levels
  res@cnMinLevelValF       =   -5.25                 ; set min contour level
  res@cnMaxLevelValF       =   5.25                  ; set max contour level
  res@cnLevelSpacingF      =   0.5                  ; set contour spacing

  plot = gsn_csm_contour_map(wks, vryrgseason(:,:), res)

  delete(vryrgseason)
  print("Current year is: "+i)
end do

end
