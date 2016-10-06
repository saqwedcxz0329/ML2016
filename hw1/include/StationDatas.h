#ifndef STATIONDATAS_H
#define STATIONDATAS_H


class StationDatas
{
    public:
        StationDatas();
        virtual ~StationDatas();

        double* getAmb_temp() { return amb_temp; }
        void setAmb_temp(double* val) { amb_temp = val; }
        double* getCh4() { return ch4; }
        void setCh4(double* val) { ch4 = val; }
        double* getCo() { return co; }
        void setCo(double* val) { co = val; }
        double* getNmhc() { return nmhc; }
        void setNmhc(double* val) { nmhc = val; }
        double* getNo() { return no; }
        void setNo(double* val) { no = val; }
        double* getNo2() { return no2; }
        void setNo2(double* val) { no2 = val; }
        double* getNox() { return nox; }
        void setNox(double* val) { nox = val; }
        double* getO3() { return o3; }
        void setO3(double* val) { o3 = val; }
        double* getPm10() { return pm10; }
        void setPm10(double* val) { pm10 = val; }
        double* getPm_two_point_five() { return pm_two_point_five; }
        void setPm_two_point_five(double* val) { pm_two_point_five = val; }
        double* getRainfall() { return rainfall; }
        void setRainfall(double* val) { rainfall = val; }
        double* getRh() { return rh; }
        void setRh(double* val) { rh = val; }
        double* getSo2() { return so2; }
        void setSo2(double* val) { so2 = val; }
        double* gettHc() { return thc; }
        void settHc(double* val) { thc = val; }
        double* getWd_hr() { return wd_hr; }
        void setWd_hr(double* val) { wd_hr = val; }
        double* getWind_direc() { return wind_direc; }
        void setWind_direc(double* val) { wind_direc = val; }
        double* getWind_speed() { return wind_speed; }
        void setWind_speed(double* val) { wind_speed = val; }
        double* getWs_hr() { return ws_hr; }
        void setWs_hr(double* val) { ws_hr = val; }

    private:
        double* amb_temp;
        double* ch4;
        double* co;
        double* nmhc;
        double* no;
        double* no2;
        double* nox;
        double* o3;
        double* pm10;
        double* pm_two_point_five;
        double* rainfall;
        double* rh;
        double* so2;
        double* thc;
        double* wd_hr;
        double* wind_direc;
        double* wind_speed;
        double* ws_hr;
};

#endif // STATIONDATAS_H
