import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'home_page.dart';

class LoginPage extends StatefulWidget {
  @override
  _LoginPageState createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _emailController = TextEditingController();
  bool _isLoading = false;

  final String _tempPassword = 'TempPassword123!'; // 회원가입 때 사용한 임시 비밀번호

  Future<void> _login() async {
    final email = _emailController.text.trim();

    if (email.isEmpty) {
      _showSnack('이메일을 입력해주세요.');
      return;
    }

    setState(() => _isLoading = true);

    try {
      // FirebaseAuth 로그인 시도
      final userCredential = await FirebaseAuth.instance
          .signInWithEmailAndPassword(email: email, password: _tempPassword);

      final user = userCredential.user;

      if (user == null || !user.emailVerified) {
        _showSnack('이메일 인증을 완료한 계정으로 로그인해주세요.');
        return;
      }

      // Firestore에서 사용자 이름 불러오기
      final userDoc =
          await FirebaseFirestore.instance
              .collection('users')
              .doc(user.uid)
              .get();

      if (!userDoc.exists) {
        _showSnack('회원 정보를 찾을 수 없습니다.');
        return;
      }

      final userName = userDoc['name'] ?? '사용자';

      _showSnack('로그인 성공!');
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => HomePage(userName: userName)),
      );
    } on FirebaseAuthException catch (e) {
      if (e.code == 'user-not-found') {
        _showSnack('가입된 계정을 찾을 수 없습니다.');
      } else if (e.code == 'wrong-password') {
        _showSnack('비밀번호가 잘못되었습니다.');
      } else {
        _showSnack('로그인 오류: ${e.message}');
      }
    } catch (e) {
      _showSnack('알 수 없는 오류가 발생했습니다.');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _showSnack(String message) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  void dispose() {
    _emailController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF2C2B34),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: IconThemeData(color: Colors.white),
        title: SizedBox.shrink(),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            TextField(
              cursorColor: Colors.grey,
              controller: _emailController,
              style: TextStyle(color: Colors.white),
              decoration: InputDecoration(
                labelText: '이메일',
                labelStyle: TextStyle(color: Colors.white70),
                filled: true,
                fillColor: Colors.white24,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(30),
                  borderSide: BorderSide.none,
                ),
              ),
              keyboardType: TextInputType.emailAddress,
            ),
            SizedBox(height: 30),
            ElevatedButton(
              onPressed: _isLoading ? null : _login,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: Colors.black,
                minimumSize: Size(double.infinity, 48),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
              ),
              child:
                  _isLoading
                      ? CircularProgressIndicator(color: Colors.black)
                      : Text('로그인'),
            ),
          ],
        ),
      ),
    );
  }
}
