import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'dart:async';

class SignUpPage extends StatefulWidget {
  @override
  _SignUpPageState createState() => _SignUpPageState();
}

class _SignUpPageState extends State<SignUpPage> {
  final _nameController = TextEditingController();
  final _birthController = TextEditingController();
  final _phoneController = TextEditingController();
  final _nicknameController = TextEditingController();
  final _emailController = TextEditingController();

  bool _emailSent = false;
  bool _isAuthVerified = false;
  bool _isSendingEmail = false;
  Timer? _emailCheckTimer;
  String? _gender;

  String _tempPassword = "TempPassword123!";

  void _sendEmailVerification() async {
    final email = _emailController.text.trim();

    if (email.isEmpty) {
      _showSnack('이메일을 입력해주세요.');
      return;
    }

    setState(() => _isSendingEmail = true);

    try {
      final methods = await FirebaseAuth.instance.fetchSignInMethodsForEmail(
        email,
      );

      if (methods.isNotEmpty) {
        _showSnack('이미 가입된 이메일입니다. 로그인하거나 다른 이메일을 입력하세요.');
        setState(() => _isSendingEmail = false);
        return;
      }

      final userCredential = await FirebaseAuth.instance
          .createUserWithEmailAndPassword(
            email: email,
            password: _tempPassword,
          );

      final user = userCredential.user!;
      await user.sendEmailVerification();

      setState(() {
        _emailSent = true;
        _isSendingEmail = false;
      });

      _startEmailVerificationCheck();

      _showSnack('인증 이메일이 전송되었습니다. 이메일을 확인해주세요.');
    } on FirebaseAuthException catch (e) {
      _showSnack('이메일 전송 오류: ${e.message}');
      setState(() => _isSendingEmail = false);
    } catch (e) {
      _showSnack('알 수 없는 오류가 발생했습니다.');
      setState(() => _isSendingEmail = false);
    }
  }

  void _startEmailVerificationCheck() {
    _emailCheckTimer?.cancel();
    _emailCheckTimer = Timer.periodic(Duration(seconds: 3), (timer) async {
      final user = FirebaseAuth.instance.currentUser;
      await user?.reload();
      if (user != null && user.emailVerified) {
        timer.cancel();
        setState(() {
          _isAuthVerified = true;
        });
        _showSnack('이메일 인증 완료!');
      }
    });
  }

  Future<void> _saveUserData() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    try {
      await FirebaseFirestore.instance.collection('users').doc(user.uid).set({
        'name': _nameController.text.trim(),
        'email': user.email,
        'gender': _gender ?? '',
        'birth': _birthController.text.trim(),
        'phone': _phoneController.text.trim(),
        'nickname': _nicknameController.text.trim(),
        'createdAt': FieldValue.serverTimestamp(),
      });
    } catch (e) {
      _showSnack('사용자 정보 저장 중 오류가 발생했습니다.');
    }
  }

  void _completeSignUp() async {
    if (_nameController.text.trim().isEmpty ||
        _gender == null ||
        _birthController.text.trim().length != 8 ||
        _emailController.text.trim().isEmpty ||
        _phoneController.text.trim().length < 10 ||
        _nicknameController.text.trim().isEmpty ||
        !_isAuthVerified) {
      _showSnack('모든 필드를 올바르게 입력하고 이메일 인증을 완료해주세요.');
      return;
    }

    await _saveUserData();

    showDialog(
      context: context,
      builder:
          (_) => AlertDialog(
            backgroundColor: Colors.white,
            title: Text('회원가입 완료!'),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.of(context).popUntil((route) => route.isFirst);
                },
                child: Text('확인'),
              ),
            ],
          ),
    );
  }

  void _showSnack(String message) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  void dispose() {
    _nameController.dispose();
    _birthController.dispose();
    _phoneController.dispose();
    _nicknameController.dispose();
    _emailController.dispose();
    _emailCheckTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF1A171D),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text('회원가입', style: TextStyle(fontWeight: FontWeight.bold)),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildInputField('이름', _nameController),
              SizedBox(height: 16),
              Text('성별', style: TextStyle(color: Colors.white)),
              Row(
                children:
                    ['남', '여'].map((g) {
                      final selected = _gender == g;
                      return Expanded(
                        child: GestureDetector(
                          onTap: () => setState(() => _gender = g),
                          child: Container(
                            margin: EdgeInsets.symmetric(
                              horizontal: 5,
                              vertical: 10,
                            ),
                            padding: EdgeInsets.symmetric(vertical: 12),
                            decoration: BoxDecoration(
                              color: selected ? Colors.white : Colors.white24,
                              borderRadius: BorderRadius.circular(30),
                            ),
                            alignment: Alignment.center,
                            child: Text(
                              g,
                              style: TextStyle(
                                color: selected ? Colors.black : Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                      );
                    }).toList(),
              ),
              _buildInputField('생년월일 (예: 19900101)', _birthController),
              SizedBox(height: 16),
              _buildInputField('이메일', _emailController),
              SizedBox(height: 10),
              ElevatedButton(
                onPressed: _isSendingEmail ? null : _sendEmailVerification,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: Colors.black,
                  padding: EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
                child:
                    _isSendingEmail
                        ? CircularProgressIndicator(color: Colors.black)
                        : Text('인증 이메일 발송'),
              ),
              if (_emailSent && !_isAuthVerified)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 10),
                  child: Text(
                    '이메일 인증을 완료해주세요.',
                    style: TextStyle(color: Colors.white70),
                  ),
                ),
              if (_isAuthVerified)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 10),
                  child: Text(
                    '✅ 이메일 인증 완료',
                    style: TextStyle(color: Colors.greenAccent),
                  ),
                ),
              _buildInputField('전화번호', _phoneController),
              SizedBox(height: 16),
              _buildInputField('닉네임', _nicknameController),
              SizedBox(height: 30),
              ElevatedButton(
                onPressed: _completeSignUp,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: Colors.black,
                  minimumSize: Size(double.infinity, 48),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
                child: Text('회원가입 완료'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInputField(String label, TextEditingController controller) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: TextStyle(color: Colors.white)),
        SizedBox(height: 8),
        TextField(
          cursorColor: Colors.black,
          controller: controller,
          style: TextStyle(color: Colors.white),
          decoration: InputDecoration(
            filled: true,
            fillColor: Colors.white24,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(30),
              borderSide: BorderSide.none,
            ),
            contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          ),
        ),
      ],
    );
  }
}
