import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

Future<void> saveDiagnosisResult({
  required String carId,
  required double airPct,
  required String status,
  required String wheelPosition,
  required String comment,
}) async {
  final user = FirebaseAuth.instance.currentUser;
  if (user == null) {
    print('로그인된 사용자가 없습니다.');
    return;
  }

  try {
    await FirebaseFirestore.instance
        .collection('users')
        .doc(user.uid)
        .collection('cars')
        .doc(carId)
        .collection('air')
        .add({
          'air_pct': airPct,
          'status': status,
          'wheelPosition': wheelPosition,
          'comment': comment,
          'createdAt': FieldValue.serverTimestamp(),
        });

    print('✅ Firestore 저장 완료');
  } catch (e) {
    print('❌ Firestore 저장 실패: $e');
  }
}

Future<void> saveCrackDiagnosisResult({
  required String carId,
  required double riskScore,
  required String status,
  required String wheelPosition,
  required String comment,
  required String imageUrl,
}) async {
  final user = FirebaseAuth.instance.currentUser;
  if (user == null) return;

  try {
    await FirebaseFirestore.instance
        .collection('users')
        .doc(user.uid)
        .collection('cars')
        .doc(carId)
        .collection('crack')
        .add({
          'risk_score': riskScore,
          'status': status,
          'wheelPosition': wheelPosition,
          'comment': comment,
          'imageUrl': imageUrl,
          'createdAt': FieldValue.serverTimestamp(),
        });
    print("✅ crack 진단 결과 저장 완료");
  } catch (e) {
    print("❌ crack 진단 저장 실패: $e");
  }
}
