package MTN;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

//更新顺序U,S,T,a,b,c


public class MTN_TRL extends InitTensor{

	public double sumTime = 0; //训练累计时间

	public int tr = 0;  //统计本轮迭代与前一轮小于误差的次数

	public int threshold = 0;  //连续下降轮数小于误差范围切达到阈值终止训练

	public boolean flagRMSE = true, flagMAE = true;

	public String str = null;

	public double yita = 0;
	public double gama = 0;
	public double lambda = 0;  //因子矩阵正则化参数
	public double lambda_b = 0;  //线性偏差正则化参数

	public double maxRR = -100;
	public int minRRRound = 0;


	public int population = 10;
	public double f;


	//	public int population=10;
	public double[] maxX = {1, 12, 8};
	public double[] minX = {0, 8, 4};
	public double[][] px;
	public double[] timeCost = new double[500];


	public void initPopulation()
	{
		px = new double[population][3];
		Random random = new Random();
		for (int p = 0; p < population; p++) {
			for (int h = 0; h < 3 ; h++)
			{
				px[p][h] = minX[h] + random.nextDouble() * (maxX[h]- minX[h]);
			}
		}
	}


	MTN_TRL(String trainFile, String validFile, String testFile, String separator )
	{
		super(trainFile, validFile, testFile, separator);
	}

	public void train(FileWriter Parameters) throws IOException
	{
		long startTime = System.currentTimeMillis();   //记录开始训练时间

//		FileWriter  fw = new FileWriter(new File(trainFile.replace(".txt", "_")+rank+"_"+lambda+"_"+
//		                lambda_b+"_"+new Date().getTime() / 1000+"_BNLFT.txt"));
//
//		fw.write("round :: everyRoundRMSE :: everyRoundMAE :: costTime(ms) \n");
//		fw.flush();

		initFactorMatrix();
		initAssistMatrix();
//		initSliceSet();
		initPopulation();

//		System.out.println("maxAID maxBID maxCID "+maxAID+" "+maxBID+" "+maxCID);
//		System.out.println("minAID minBID minCID "+minAID+" "+minBID+" "+minCID);
//		System.out.println("trainCount validCount testCount "+trainCount+" "+validCount+" "+testCount);
//		System.out.println("初始范围:"+initscale);
//		System.out.println("lambda  lambda_b: "+lambda+" "+lambda_b);


		//计算初始适应度值
		double init_rmse = 0, init_mae = 0;
		for (TensorTuple validTuple : validData) {
			// 获得元素的预测值
			validTuple.valueHat = this.getPrediction(validTuple.aID, validTuple.bID, validTuple.cID);
			init_rmse += Math.pow(validTuple.value - validTuple.valueHat, 2);
			init_mae += Math.abs(validTuple.value - validTuple.valueHat);
		}
		double init_fitness = 0.5 * Math.sqrt(init_rmse / validCount) + 0.5 * (init_mae / validCount);
//		System.out.println(init_fitness);



		double gBest = Integer.MIN_VALUE;
		double[] gBest_value = new double[3];

		for(int round = 1; round <= trainRound; round++) {
			long startRoundTime = System.currentTimeMillis();    //记录每轮的训练时间
			double[] fitness = new double[population];
			for (int p = 0; p < population; p++) {

				for (TensorTuple trainTuple : trainData) {
					trainTuple.valueHat = this.getPrediction(trainTuple.aID, trainTuple.bID, trainTuple.cID);

					for (int r1 = 1; r1 <= rank; r1++) {
						for (int r4 = 1; r4 <= rank1; r4++) {
							double temp = 0;
							for (int r2 = 1; r2 <= rank; r2++) {
								for (int r3 = 1; r3 <= rank; r3++) {
									temp += G[r1][r2][r3] * D[trainTuple.bID][r2][r4] * T[trainTuple.cID][r3];
								}
							}
							Sup[trainTuple.aID][r1][r4] = (trainTuple.value - trainTuple.valueHat) * temp - Math.pow(2, -px[p][2]) * S[trainTuple.aID][r1][r4];
							S[trainTuple.aID][r1][r4] += Math.pow(2, -px[p][1]) * Sup[trainTuple.aID][r1][r4] - px[p][0] * Sdown[trainTuple.aID][r1][r4];
							Sdown[trainTuple.aID][r1][r4] = px[p][0] * Sdown[trainTuple.aID][r1][r4] - Math.pow(2, -px[p][1]) * Sup[trainTuple.aID][r1][r4];
						}
					}
				}


				for (TensorTuple trainTuple : trainData) {
					trainTuple.valueHat = this.getPrediction(trainTuple.aID, trainTuple.bID, trainTuple.cID);

					for (int r2 = 1; r2 <= rank; r2++) {
						for (int r4 = 1; r4 <= rank1; r4++) {
							double temp = 0;
							for (int r1 = 1; r1 <= rank; r1++) {
								for (int r3 = 1; r3 <= rank; r3++) {
									temp += G[r1][r2][r3] * S[trainTuple.aID][r1][r4] * T[trainTuple.cID][r3];
								}
							}
							Dup[trainTuple.bID][r2][r4] = (trainTuple.value - trainTuple.valueHat) * temp - Math.pow(2, -px[p][2]) * D[trainTuple.bID][r2][r4];
							D[trainTuple.bID][r2][r4] += Math.pow(2, -px[p][1]) * Dup[trainTuple.bID][r2][r4] - px[p][0] * Ddown[trainTuple.bID][r2][r4];
							Ddown[trainTuple.bID][r2][r4] = px[p][0] * Ddown[trainTuple.bID][r2][r4] - Math.pow(2, -px[p][1]) * Dup[trainTuple.bID][r2][r4];
						}
					}
				}


				for (TensorTuple trainTuple : trainData) {
					trainTuple.valueHat = this.getPrediction(trainTuple.aID, trainTuple.bID, trainTuple.cID);

					for (int r3 = 1; r3 <= rank; r3++) {
						double temp = 0;
						for (int r1 = 1; r1 <= rank; r1++) {
							for (int r2 = 1; r2 <= rank; r2++) {
								for (int r4 = 1; r4 <= rank1; r4++) {
									temp += G[r1][r2][r3] * D[trainTuple.bID][r2][r4] * S[trainTuple.aID][r1][r4];
								}
							}
						}
						Tup[trainTuple.cID][r3] = (trainTuple.value - trainTuple.valueHat) * temp - Math.pow(2, -px[p][2]) * T[trainTuple.cID][r3];
						T[trainTuple.cID][r3] += Math.pow(2, -px[p][1]) * Tup[trainTuple.cID][r3] - px[p][0] * Tdown[trainTuple.cID][r3];
						Tdown[trainTuple.cID][r3] = px[p][0] * Tdown[trainTuple.cID][r3] - Math.pow(2, -px[p][1]) * Tup[trainTuple.cID][r3];
					}
				}


				for (TensorTuple trainTuple : trainData) {
					trainTuple.valueHat = this.getPrediction(trainTuple.aID, trainTuple.bID, trainTuple.cID);

					for (int r1 = 1; r1 <= rank; r1++) {
						for (int r2 = 1; r2 <= rank; r2++) {
							for (int r3 = 1; r3 <= rank; r3++) {
								double temp = 0;
								for (int r4 = 1; r4 <= rank1; r4++) {
									temp += S[trainTuple.aID][r1][r4] * D[trainTuple.bID][r2][r4] * T[trainTuple.cID][r3];
								}
								Gup[r1][r2][r3] = (trainTuple.value - trainTuple.valueHat) * temp - Math.pow(2, -px[p][2]) * G[r1][r2][r3];
								G[r1][r2][r3] += Math.pow(2, -px[p][1]) * Gup[r1][r2][r3] - px[p][0] * Gdown[r1][r2][r3];
								Gdown[r1][r2][r3] = px[p][0] * Gdown[r1][r2][r3] - Math.pow(2, -px[p][1]) * Gup[r1][r2][r3];
							}
						}
					}
				}


//			bias更新

				for (TensorTuple trainTuple : trainData) {
					trainTuple.valueHat = this.getPrediction(trainTuple.aID, trainTuple.bID, trainTuple.cID);
					aup[trainTuple.aID] = (trainTuple.value - trainTuple.valueHat) - Math.pow(2, -px[p][2])*a[trainTuple.aID];
					a[trainTuple.aID] += Math.pow(2, -px[p][1]) * aup[trainTuple.aID] - px[p][0] * adown[trainTuple.aID];
					adown[trainTuple.aID] = px[p][0] * adown[trainTuple.aID] - Math.pow(2, -px[p][1]) * aup[trainTuple.aID];
				}


				for (TensorTuple trainTuple : trainData) {
					trainTuple.valueHat = this.getPrediction(trainTuple.aID, trainTuple.bID, trainTuple.cID);
					bup[trainTuple.bID] = (trainTuple.value - trainTuple.valueHat) - Math.pow(2, -px[p][2])*b[trainTuple.bID];
					b[trainTuple.bID] += Math.pow(2, -px[p][1]) * bup[trainTuple.bID] - px[p][0] * bdown[trainTuple.bID];
					bdown[trainTuple.bID] = px[p][0] * bdown[trainTuple.bID] - Math.pow(2, -px[p][1]) * bup[trainTuple.bID];
				}


				for (TensorTuple trainTuple : trainData) {
					trainTuple.valueHat = this.getPrediction(trainTuple.aID, trainTuple.bID, trainTuple.cID);
					cup[trainTuple.cID] = (trainTuple.value - trainTuple.valueHat) - Math.pow(2, -px[p][2])*c[trainTuple.cID];
					c[trainTuple.cID] += Math.pow(2, -px[p][1]) * cup[trainTuple.cID] - px[p][0] * cdown[trainTuple.cID];
					cdown[trainTuple.cID] = px[p][0] * cdown[trainTuple.cID] - Math.pow(2, -px[p][1]) * cup[trainTuple.cID];
				}


				double square1 = 0, absCount1 = 0;
				for (TensorTuple validTuple : validData) {
					// 获得元素的预测值
					validTuple.valueHat = this.getPrediction(validTuple.aID, validTuple.bID, validTuple.cID);
					square1 += Math.pow(validTuple.value - validTuple.valueHat, 2);
					absCount1 += Math.abs(validTuple.value - validTuple.valueHat);
				}
				fitness[p] = 0.5 * Math.sqrt(square1 / validCount) + 0.5 * (absCount1 / validCount);
//				System.out.println(fitness[p]);
			}


			// 计算相对提升
			double[] Fitness = new double[population];
			double Fdown = fitness[population - 1] - init_fitness;
			for (int p = 0; p < population; p++) {
				if (p == 0) {
					Fitness[p] = (fitness[p] - init_fitness) / Fdown;
				} else Fitness[p] = (fitness[p] - fitness[p - 1]) / Fdown;

			}
			init_fitness = fitness[population - 1];

			//找到全局最优
			for (int p = 0; p < population; p++) {
				if (Fitness[p] > gBest) {
					gBest = Fitness[p];
					for (int h = 0; h < 3; h++) {
						gBest_value[h] = px[p][h];
					}
				}
			}


			double[] fitness_sort = new double[this.population];
			double[][] new_px = new double[this.population][3];
			System.arraycopy(Fitness, 0, fitness_sort, 0, this.population);
			//升序排序
			Arrays.sort(fitness_sort);

//			System.out.println(Fitness);
//			System.out.println(fitness_sort);
//			for (int k = 0; k <= fitness_sort.length; k++)
//			{
//				System.out.println(Fitness[k]);
//				System.out.println(fitness_sort[k]);
//			}

			for (int p = 0; p < population; p++) {
				int temp = get_index(Fitness, fitness_sort[p]);
//				System.out.println(temp);
//				System.out.println(init_fitness);
				for (int h = 0; h < 3; h++) {
					new_px[population - 1 - p][h] = px[temp][h];
				}
			}


			for (int p = population / 2; p < population; p++) {
				Random random = new Random();
				int r1 = random.nextInt(population / 2);
				int r2 = random.nextInt(population / 2);
				int r3 = random.nextInt(3);
//				double f = 0.01 * random.nextInt(100);
				for (int h = 0; h < 3; h++) {
					if (random.nextDouble() <= 0.3 || h == r3) {
						new_px[p][h] = gBest_value[h] + f * (new_px[r1][h] - new_px[r2][h]);
					}
//
					if (new_px[p][h] < minX[h]) {
						new_px[p][h] = minX[h];
					}
					if (new_px[p][h] > maxX[h]) {
						new_px[p][h] = maxX[h];
					}
				}
			}
			for (int p = 0; p < population / 2; p++) {
				for (int h = 0; h < 3; h++) {
					px[p][h] = new_px[p][h];
				}
			}

			timeCost[0] = 0;
			timeCost[round] = timeCost[round-1] + (System.currentTimeMillis() - startRoundTime)/1000.00;

			// 每一轮参数更新后，开始对验证集测试
			double square = 0, absCount = 0;
			for (TensorTuple validTuple : validData) {
				// 获得元素的预测值
				validTuple.valueHat = this.getPrediction(validTuple.aID, validTuple.bID, validTuple.cID);
				square += Math.pow(validTuple.value - validTuple.valueHat, 2);
				absCount += Math.abs(validTuple.value - validTuple.valueHat);

			}

			everyRoundRMSE[round] = Math.sqrt(square / validCount);
			everyRoundMAE[round] = absCount / validCount;

//			long endRoundTime = System.currentTimeMillis();
//			sumTime += (endRoundTime - startRoundTime);

			// 每一轮参数更新后，记录测试集结果
			double square2 = 0, absCount2 = 0, absCount33 = 0, absCount44 = 0;
			for (TensorTuple testTuple : testData) {
				// 获得元素的预测值
				testTuple.valueHat = this.getPrediction(testTuple.aID, testTuple.bID, testTuple.cID);
				square2 += Math.pow(testTuple.value - testTuple.valueHat, 2);
				absCount2 += Math.abs(testTuple.value - testTuple.valueHat);
//				absCount33 += (Math.abs(testTuple.value - testTuple.valueHat)) / (testTuple.value+0.01);
				absCount44 += Math.pow(testSum / testCount - testTuple.value, 2);
			}

			everyRoundRMSE2[round] = Math.sqrt(square2 / testCount);
			everyRoundMAE2[round] = absCount2 / testCount;
//			everyRoundMAPE2[round] = 100 * absCount33 / testCount;
			everyRoundRR2[round] = 1 - square2 / absCount44;

			System.out.println(everyRoundRMSE2[round] + " " + everyRoundMAE2[round] + " " + everyRoundRR2[round]);

//			fw.write(round + "::" + everyRoundRMSE[round] + "::" + everyRoundMAE[round]
//					+"::"+ (endRoundTime-startRoundTime)+"::"+ everyRoundRMSE2[round] + "::" + everyRoundMAE2[round]+"\n");
//			fw.flush();

			if (maxRR < everyRoundRR2[round]) {
				maxRR = everyRoundRR2[round];
				minRRRound = round;
			}

			if (everyRoundRMSE[round - 1] - everyRoundRMSE[round] > errorgap) {
				if (minRMSE > everyRoundRMSE[round]) {
					minRMSE = everyRoundRMSE[round];
					minRMSERound = round;
				}

				flagRMSE = false;
				tr = 0;
			}

			if (everyRoundMAE[round - 1] - everyRoundMAE[round] > errorgap) {
				if (minMAE > everyRoundMAE[round]) {
					minMAE = everyRoundMAE[round];
					minMAERound = round;
				}

				flagMAE = false;
				tr = 0;
			}

			if (flagRMSE && flagMAE) {
				tr++;
				if (tr == threshold)
					break;
			}

			flagRMSE = true;
			flagMAE = true;

		}

		long endTime = System.currentTimeMillis();

//		fw.write("总训练时间："+(endTime-startTime)/1000+"s\n");
//		fw.flush();
//		fw.write("validation minRMSE:"+minRMSE+"  minRSMERound"+minRMSERound+"\n");
//		fw.write("validation minMAE:"+minMAE+"  minMAERound"+minMAERound+"\n");
//		fw.write("testing minRMSE:"+everyRoundRMSE2[minRMSERound]+"  minRSMERound"+minRMSERound+"\n");
//		fw.write("testing minMAE:"+everyRoundMAE2[minMAERound]+"  minMAERound"+minMAERound+"\n");
//		fw.flush();
//		fw.write("rank="+rank+"\n");
//		fw.flush();
//		fw.write("trainCount: "+trainCount+"validCount: "+validCount+"   testCount: "+testCount+"\n");
//		fw.flush();
//        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
//        fw.write("训练时间："+df.format(new Date())+"\n");// new Date()为获取当前系统时间
//        fw.flush();
//		fw.write("初始范围"+initscale+"\n");
//		fw.flush();
//		fw.write("maxAID maxBID maxCID "+maxAID+" "+maxBID+" "+maxCID+"\n");
//		fw.write("minAID minBID minCID "+minAID+" "+minBID+" "+minCID+"\n");
//		fw.close();

		Parameters.write("testing minRMSE:" + everyRoundRMSE2[minRMSERound] + " minRSMERound::" + minRMSERound + " RMSECostTime::" +timeCost[minRMSERound]/60+"min\n");
		Parameters.write("testing minMAE:" + everyRoundMAE2[minMAERound]+ " minMAERound::" + minMAERound + " MAECostTime::"+timeCost[minMAERound]/60+"min\n");
//		Parameters.write("testing minMAPE:" + everyRoundMAPE2[minMAPERound]+ " minMAPERound::" + minMAPERound + " MAPECostTime::" + "\n");
		Parameters.write("testing minRR:" + everyRoundRR2[minRRRound]+ " minRRRound::" + minRRRound + " RRCostTime::"+timeCost[minRMSERound]/60+"min\n");
		Parameters.write( "init:"+ initscale+"--"+initscale2+"yita: "+ yita + "::lambda: "+ lambda +"training time::"+ timeCost[Math.max(minRMSERound,minMAERound)]/60+ "min\n");
		Parameters.write( "------------------------------------------------------------------------------------------------" + "\n");
		Parameters.flush();

		System.out.println("***********************************************");
		System.out.println("rank::"+this.rank+"\n");
//		System.out.println("validation minRMSE:"+minRMSE+"  minRSMERound"+minRMSERound);
//		System.out.println("validation minMAE:"+minMAE+"  minMAERound"+minMAERound);
		System.out.println("总训练时间:"+(endTime-startTime)/60000.00+"min\n");
		System.out.println("训练时间:"+ timeCost[Math.max(minRMSERound,minMAERound)]/60000.00+"min\n");
		System.out.println("testing minRMSE:"+everyRoundRMSE2[minRMSERound]+"  minRSMERound"+minRMSERound);
		System.out.println("testing minMAE:"+everyRoundMAE2[minMAERound]+"  minMAERound"+minMAERound);


		for (int i = 1; i < maxAID+1; i++) {
			for (int j = 1; j < maxBID+1; j++) {
				for (int k = 1; k < maxCID+1; k++) {
					double value = getPrediction(i,j,k);
					Parameters.write(i+"::"+j+"::"+k+"::"+value+"\n");
				}
			}
		}

	}


	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		for (int number = 9; number <= 9; number++) {

			long S = System.currentTimeMillis();

//			FileWriter Parameters = new FileWriter(new File("D:\\Tensor\\R=5\\CTTN-ab\\test" + number + ".txt"));
			FileWriter Parameters = new FileWriter(new File("D:\\Tensor\\R=5\\MTN-D9-10\\MTN\\test1" + number + ".txt"));
			for (int i = 0; i <= 0; i++) {
				for (int j = 0; j <= 0; j++) {
					//训练：验证：测试=5%：5%：90%
					MTN_TRL bnlft = new MTN_TRL(
//							"D:\\Tensor\\Dataset\\network\\D" + number + "\\0.1tr.txt",
//							"D:\\Tensor\\Dataset\\network\\D" + number + "\\0.2va.txt",
//							"D:\\Tensor\\Dataset\\network\\D" + number + "\\0.7te.txt", "::");
//							"D:\\Tensor\\Dataset\\network\\case2\\0." + number + "\\D4-tr.txt",
//							"D:\\Tensor\\Dataset\\network\\case2\\0." + number + "\\D4-va.txt",
//							"D:\\Tensor\\Dataset\\network\\case2\\0." + number + "\\D4-te.txt", "::");
							"D:\\Tensor\\Dataset\\traffic\\shanghai_speed\\tr.txt",
							"D:\\Tensor\\Dataset\\traffic\\shanghai_speed\\va.txt",
							"D:\\Tensor\\Dataset\\traffic\\shanghai_speed\\te.txt", "::");

//					if (number <= 4) {
//						bnlft.initscale = 0.14;
//						bnlft.initscale2 = 0.04;
//					}
//					else {
//						bnlft.initscale = 0.3;
//						bnlft.initscale2 = 0.07;
//					}
					bnlft.initscale = 0.08;
					bnlft.initscale2 = 0.04;

					bnlft.threshold = 2;
					bnlft.rank = 5;
					bnlft.rank1 = 5;
					bnlft.trainRound = 500;
					bnlft.errorgap = 1E-5;
//					bnlft.yita = Math.pow(2, (-10));
//					bnlft.gama = 0.9;
//					bnlft.lambda = Math.pow(2, (-11));     //根据数据集设置
//					bnlft.lambda_b = bnlft.lambda;   //根据数据集设置
					bnlft.population = 10;
					bnlft.f = 0.9;

					try {
						bnlft.initData(bnlft.trainFile, bnlft.trainData, 1);
						bnlft.initData(bnlft.validFile, bnlft.validData, 2);
						bnlft.initData(bnlft.testFile, bnlft.testData, 3);
						bnlft.train(Parameters);
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
			long E = System.currentTimeMillis();
			Parameters.write( "TotalTime is: " + (E - S)/60 + "\n");
			Parameters.flush();
		}
	}
}

