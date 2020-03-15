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
  dir3   = "/scratch/d/djones/harryli/obsdataset/cru/"
  dir4   = "/scratch/d/djones/harryli/obsdataset/GPCP/"
  dir5   = "/scratch/d/djones/harryli/obsdataset/GPCC/"
  rdir   = "/scratch/d/djones/harryli/cesm1.2/inputdata/atm/cam/dst/"
  f1     = addfile (dir1+"fv05_AMIP_VRcesm_diag_1992_2005.cam.h0.1992-2004.nc", "r")
  f2     = addfile (dir2+"prect_AMIPdiag_1920_today_cesm122_defaultini.cam.h0.199001-200611.nc", "r")
  f3     = addfile (dir3+"cru_ts3.21.1901.2012.pre_orig_mmday.nc","r")
  f4     = addfile (dir4+"gpcp_cdr_v23rB1_197901-201608.nc","r")
  f5     = addfile (dir5+"precip.mon.total.v7.nc","r")
  rf     = addfile (rdir+"dst_1.9x2.5_c090203.nc","r")

  print("Start loading data...")
  vrm1  = f1->PRECT(36:155,:,:)
  vrm2  = f2->PRECT(60:179,:,:)
  vrm3  = f3->pre(1128:1247,:,:)
  vrm4  = f4->precip(192:311,:,:)
  vrm5  = f5->precip(1128:1247,:,:)
  print(vrm5(0,:,0))
  print("Loading data:Done!")

  lon1    = f1->lon
  lat1    = f1->lat
  lon2    = f2->lon
  lat2    = f2->lat
  lon3    = f3->lon
  lat3    = f3->lat
  lon4    = f4->longitude
  lat4    = f4->latitude
  lon5    = f5->lon
  lat5    = f5->lat
  rlon    = rf->lon
  rlat    = rf->lat
 
  print(lat5)

  iniyear = 1995
  endyear = 2004
  yeart = ispan(1,10,1)
  montht = ispan(1,12,1)
  seasont = ispan(1,40,1)
  m=(/01,02,03,04,05,06,07,08,09,10,11,12/)
  mdays=(/31,28,31,30,31,30,31,31,30,31,30,31/)
  month = (/"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"/)
  legend1 = " VRseasia"
  legend2 = " cesm1_2_2"
  legend3 = " CRU"
  legend4 = " GPCP"
  legend5 = " GPCC v7"
;*****************************************************
; some parameters
;***************************************************** 
  nyrs   = 5
  dsizes = dimsizes(vrm2)
  ntime  = dsizes(0)
  nlat   = dimsizes(rlat)
  nlon   = dimsizes(rlon)
  ndata  = 5
  nmons  = 12
  nseason = 4*(endyear-iniyear+1)
  nyear  = endyear-iniyear+1
;*****************************************************   
; Calculation
;*****************************************************
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

do i =1,120
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

  vrAvgLat1 = dim_avg_n_Wrap(vrm1(:,{20:28},:),1)
  vrAvgLon1 = dim_avg_n_Wrap(vrAvgLat1(:,{76:87}),1)
  vrmon1    = vrAvgLon1

  vrAvgLat2 = dim_avg_n_Wrap(vrm2(:,{20:28},:),1)
  vrAvgLon2 = dim_avg_n_Wrap(vrAvgLat2(:,{76:87}),1)
  vrmon2    = vrAvgLon2

  vrAvgLat3 = dim_avg_n_Wrap(vrm3(:,{20:28},:),1)
  vrAvgLon3 = dim_avg_n_Wrap(vrAvgLat3(:,{76:87}),1)
  vrmon3    = vrAvgLon3

  vrAvgLat4 = dim_avg_n_Wrap(vrm4(:,{20:28},:),1)
  vrAvgLon4 = dim_avg_n_Wrap(vrAvgLat4(:,{76:87}),1)
  vrmon4    = vrAvgLon4
  
  vrAvgLat5 = dim_avg_n_Wrap(vrm5(:,{20:28},:),1)
  vrAvgLon5 = dim_avg_n_Wrap(vrAvgLat5(:,{76:87}),1)
  vrmon5    = vrAvgLon5

  vrmon1 = 86400*1000*vrmon1
  vrmon2 = 86400*1000*vrmon2

  vryr1  = new (12,double)
  vryr2  = new (12,double)
  vryr3  = new (12,double)
  vryr4  = new (12,double)
  vryr5  = new (12,double)

  k=1
  do while (k.le.12)
     vryr1(k-1) = dim_avg_n_Wrap(vrmon1(k-1::12),0)
     vryr2(k-1) = dim_avg_n_Wrap(vrmon2(k-1::12),0)
     vryr3(k-1) = dim_avg_n_Wrap(vrmon3(k-1::12),0)
     vryr4(k-1) = dim_avg_n_Wrap(vrmon4(k-1::12),0)
     vryr5(k-1) = dim_avg_n_Wrap(vrmon5(k-1::12),0)
     k=k+1
  end do

  len = new (2,integer)
  len(0)=ndata
  len(1)=nmons

  vrcom      = new(len,double)
  vrcom(0,:) = vryr1
  vrcom(1,:) = vryr2
  vrcom(2,:) = vryr3
  vrcom(3,:) = vryr4
  vrcom(4,:) = vryr5

  vrsea1 = month_to_season12(vrmon1)
  vrsea2 = month_to_season12(vrmon2)
  vrsea3 = month_to_season12(vrmon3)
  vrsea4 = month_to_season12(vrmon4)
  vrsea5 = month_to_season12(vrmon5)

  leng = new (2,integer)
  leng(0)=ndata
  leng(1)=nseason

  vrts      = new(leng,double)
  vrts(0,:) = vrsea1(1::3)
  vrts(1,:) = vrsea2(1::3)
  vrts(2,:) = vrsea3(1::3)
  vrts(3,:) = vrsea4(1::3)
  vrts(4,:) = vrsea5(1::3)
;***************************************************** 
; create plots
;***************************************************** 
  ofilename = "AMIP_VRcesm_diag_1992_2005_PRECT_Season_vsobs_1995-2004_SouAisa"
  titlename = "PRECT SouAsia climatology 1995-2004"
  wks   = gsn_open_wks ("png", ofilename)       ; open ps file

  res  = True
  res@vpXF              = 0.2
  res@vpYF              = 0.8
  res@trXMinF           = 1
  res@trXMaxF           = 12
  res@gsnDraw           = False
  res@gsnFrame          = False
  res@tiMainString      = titlename
  res@tiXAxisString     = "Month("+iniyear+"-"+endyear+")"
  res@tmXBMode          = "Explicit"
  res@tmXBValues        = ispan(1,12,1)
  res@tiYAxisString     = "PRECT (mm/day)"
  res@tmXBLabels        = (/"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"/)
  res@tiMainFontHeightF  = 0.025
  res@tiYAxisFontHeightF = 0.02

  res@xyLineThicknesses = (/2.0,2.0,2.0,2.0,2.0/)
  res@xyLineColors      = (/"blue","red","green","purple1","cyan"/)
;  res@xyLineColors      = (/"blue","red","orange"/)

  plot  = gsn_csm_xy (wks,montht,vrcom,res)

;*****************************************************
; Manually create and attach legend
;*****************************************************
  lgres                    = True
  lgres@lgLineColors       = res@xyLineColors
  lgres@lgLineThicknessF   = res@xyLineThicknesses
  lgres@lgItemType         = "Lines"
  lgres@lgLabelFontHeightF = .08
  lgres@vpWidthF           = 0.15
  lgres@vpHeightF          = 0.1
  lgres@lgPerimColor       = "orange"
  lgres@lgPerimThicknessF  = 5.0
  lbid = gsn_create_legend(wks,5,(/legend1,legend2,legend3,legend4,legend5/),lgres)

  amres = True
  amres@amParallelPosF   = 0.7
  amres@amOrthogonalPosF = 0.
  annoid1 = gsn_add_annotation(plot,lbid,amres)

  draw(plot)
  frame(wks)

;*****************************************************
; create plots
;*****************************************************
  ofilename = "AMIP_VRcesm_diag_1992_2005_PRECT_TS_vsobs_1995-2004_SouAsia"
  titlename = "PRECT SouAsia trend 1995-2004"
  wksts   = gsn_open_wks ("png", ofilename)       ; open ps file

  rests  = True
  rests@vpXF              = 0.2
  rests@vpYF              = 0.8
  rests@trXMinF           = 1
  rests@trXMaxF           = 40
  rests@gsnDraw           = False
  rests@gsnFrame          = False
  rests@tiMainString      = titlename
  rests@tiXAxisString     = "Year("+iniyear+"-"+endyear+")"
  rests@tmXBMode          = "Explicit"
  rests@tmXBValues        = ispan(2,38,4)
  rests@tiYAxisString     = "PRECT (mm/day)"
  rests@tmXBLabels        = (/"95","96","97","98","99","00","01","02","03","04"/)
  rests@tiMainFontHeightF  = 0.025
  rests@tiYAxisFontHeightF = 0.02

  rests@xyLineThicknesses = (/2.0,2.0,2.0,2.0,2.0/)
  rests@xyLineColors      = (/"blue","red","green","purple1","cyan"/)
;  rests@xyLineColors      = (/"blue","red","orange"/)

  plotts  = gsn_csm_xy (wksts,seasont,vrts,rests)

;*****************************************************
; Manually create and attach legend
;*****************************************************
  lgrests                    = True
  lgrests@lgLineColors       = res@xyLineColors
  lgrests@lgLineThicknessF   = res@xyLineThicknesses
  lgrests@lgItemType         = "Lines"
  lgrests@lgLabelFontHeightF = .08
  lgrests@vpWidthF           = 0.15
  lgrests@vpHeightF          = 0.1
  lgrests@lgPerimColor       = "orange"
  lgrests@lgPerimThicknessF  = 5.0
  lbidts = gsn_create_legend(wksts,5,(/legend1,legend2,legend3,legend4,legend5/),lgrests)

  amres = True
  amres@amParallelPosF   = 0.7
  amres@amOrthogonalPosF = 0.
  annoid1 = gsn_add_annotation(plotts,lbidts,amres)

  draw(plotts)
  frame(wksts)

end
