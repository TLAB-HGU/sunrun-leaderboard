# calib_offset_roll 필수 ablation — golab2

사용자의 규모 축소 지시에 따라 기존162항목 계획을 폐기하고 다음4조건만 실행한다. 기존 계획과 대기 상태는 superseded 파일에 보존하며 자동 재개하지 않는다.

| 조건 | 변경 | 확인할 가설 |
|---|---|---|
| baseline_s42 | 없음 | 현재 모델의 고정 기준 |
| loss_t4_sp1_nooffset_s42 | offset만 제거 | 고정 지평별 편향 보정이 이득의 원인인가 |
| loss_t4_sp0_offset_s42 | spectral 항만 제거 | 지평 결합 손실이 필요한가 |
| loss_t1_sp1_offset_s42 | 꼬리 가중치4→1 | 고속 구간의 가중 학습이 필요한가 |

현재 실행 중인baseline은 유지한다. 실제 모델학습은기준선포함3개,offset제거는동일모델예측재사용1개다. seed42만사용한다.30개특징,900rounds,24시간블록,offset336행/alpha.25,clipping,나머지하이퍼파라미터는그대로고정한다. spectral/가중치 제거 모델은각각prior와full모델및offset을자신의조건으로다시적합한다. 특징제거·창길이·계수·출력구조·추가seed·손실전체조합은실행하지않는다.

모든학습·추론·채점·통계는golab2 CPU에서CUDA_VISIBLE_DEVICES=''로실행한다. 기존작업과현재baseline tmux는보존한다. 가용CPU/RAM범위내에서최대3작업×4threads,전체CPUbudget20,메모리6GiB여유를유지한다. TUI표준launch/log/metadata/exit/score/complete형식을유지한다.

공식F1/F2/F3만평가하고학습cutoff와144h성숙조건을유지한다. 학습내3개월은offset적합에만쓰고내부선별에는쓰지않는다. 공식관측revision920588c2a716130e19b145acbbd91ace4758faf2와공식scorer를고정한다. 공식y는채점에만사용한다. baseline새학습예측과저장참조의최대절대차및S차허용오차각1e-6초과시후속작업을중단한다.

이미검증된직접증거: golab2저장모델재추론예측은원본과정확히일치하고clipping0셀이다. 동일모델의offset전후TotalS99.751183→98.762687,F1 123.340088→122.409888,F2 74.006146→74.232063,F3 60.505169→60.457317. 따라서offset은모든기간에서개선된것이아니며F1꼬리오차감소가주요후보다. 나머지두구성의기여는실험완료전확정하지않는다.

최종보고서는4조건의Total/F별S,RMSE,TRMSE,TIMSE,MAE,bias및offset구간별오차를비교한다. 시간의존성을고려한paired168h블록bootstrap2000회는추가학습없이실행한다. seed42조건부CI이며seed안정성이나독립적반복재현을입증하지않는다. 한요소씩제거한효과는비가산적일수있고,이축소설계로상호작용을확정하지않는다. 세seed가필요한강한기여판정은하지않고이번결과는기여후보의방향과크기진단으로보고한다. 성적에따른후속탐색·자동확장·기준선교체·외부제출없음.

종료조건은4조건의정상exit,예측·메타데이터·공식점수완전성및분석완료다. 모델source snapshot,git revision,파일크기·행열·시간범위·버전으로재현성을기록한다. 파일해시는workspace정책에따라계산하지않는다.

참고: XGBoost2.1 multioutput https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/multioutput.html ; custom objective diagonal Hessian guidance https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html ; Künsch1989 https://doi.org/10.1214/aos/1176347265 . 이문헌들은해당태양풍성능개선을보장하지않는다. spectral제거시gradient와Hessian기여를함께제거한다.168h/circular/F층화는사전고정분석규칙이며논문이그선택을직접검증한것은아니다.
